# Accumulate incoming PointCloud2 messages from /utlidar/cloud and republish
# the concatenated cloud on a fixed cadence (default 10 Hz), resetting the
# buffer after each publish. Optionally re-expresses the cloud in a target
# frame via a one-shot (static) TF lookup so downstream consumers (the BEV
# rasterizer) can assume a fixed sensor frame regardless of what the driver
# stamps on the input.

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener, TransformException


class AccumulatePointCloud(Node):
    def __init__(self):
        super().__init__("accumulate_pointcloud")

        self.declare_parameter("input_topic", "/utlidar/cloud")
        self.declare_parameter("output_topic", "/utlidar/cloud_accumulated")
        self.declare_parameter("publish_rate_hz", 10.0)
        # Empty target_frame disables the transform — the cloud is republished
        # in whatever frame the driver stamped on it.
        self.declare_parameter("target_frame", "")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.target_frame = self.get_parameter("target_frame").value

        self.buffer = []

        # Cached source->target transform. (R is 3x3 float64, t is 3-vector;
        # source_frame is the input cloud's frame_id at the time of the lookup.)
        self.cached_R = None
        self.cached_t = None
        self.cached_source = None
        self.tf_warned = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(PointCloud2, output_topic, 10)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.cloud_callback, qos_profile_sensor_data
        )
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.publish_accumulated)

        target_info = f" in '{self.target_frame}'" if self.target_frame else ""
        self.get_logger().info(
            f"Accumulating {input_topic} -> {output_topic}{target_info} @ {publish_rate_hz:.1f} Hz"
        )

    def cloud_callback(self, msg: PointCloud2):
        self.buffer.append(msg)

    def ensure_transform(self, source_frame: str) -> bool:
        # The radar<->utlidar_lidar TF is static, so a one-shot lookup is
        # enough; we cache the result and only re-lookup if the source frame
        # changes (it shouldn't, in practice).
        if (
            self.cached_R is not None
            and self.cached_source == source_frame
        ):
            return True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, rclpy.time.Time()
            )
        except TransformException as ex:
            if not self.tf_warned:
                self.get_logger().warn(
                    f"TF {source_frame} -> {self.target_frame} not available yet ({ex}); "
                    "passing cloud through unchanged."
                )
                self.tf_warned = True
            return False
        q = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        # Quaternion -> rotation matrix (right-handed, active rotation).
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
        t = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ], dtype=np.float64)
        self.cached_R = R
        self.cached_t = t
        self.cached_source = source_frame
        self.get_logger().info(
            f"Cached static transform {source_frame} -> {self.target_frame}"
        )
        return True

    def transform_xyz_inplace(self, data: bytearray, point_step: int, fields):
        # Locate xyz field offsets; assume float32 (PointField.FLOAT32 = 7).
        offsets = {f.name: f.offset for f in fields if f.name in ("x", "y", "z")}
        if not {"x", "y", "z"}.issubset(offsets):
            raise RuntimeError("PointCloud2 missing x/y/z fields")
        # Fast path: xyz contiguous as float32 starting at offset 0. Avoids
        # per-component gather/scatter for the common Unitree layout.
        if (
            offsets["x"] == 0
            and offsets["y"] == 4
            and offsets["z"] == 8
            and point_step >= 12
        ):
            arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, point_step)
            xyz = arr[:, 0:12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)
            xyz = xyz @ self.cached_R.T + self.cached_t
            arr[:, 0:12] = xyz.astype(np.float32).view(np.uint8).reshape(-1, 12)
            return
        # Generic path for any field layout.
        arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, point_step)
        xs = arr[:, offsets["x"]:offsets["x"] + 4].copy().view(np.float32).reshape(-1)
        ys = arr[:, offsets["y"]:offsets["y"] + 4].copy().view(np.float32).reshape(-1)
        zs = arr[:, offsets["z"]:offsets["z"] + 4].copy().view(np.float32).reshape(-1)
        pts = np.stack([xs, ys, zs], axis=1).astype(np.float64)
        pts = pts @ self.cached_R.T + self.cached_t
        pts32 = pts.astype(np.float32)
        arr[:, offsets["x"]:offsets["x"] + 4] = pts32[:, 0:1].view(np.uint8).reshape(-1, 4)
        arr[:, offsets["y"]:offsets["y"] + 4] = pts32[:, 1:2].view(np.uint8).reshape(-1, 4)
        arr[:, offsets["z"]:offsets["z"] + 4] = pts32[:, 2:3].view(np.uint8).reshape(-1, 4)

    def publish_accumulated(self):
        if not self.buffer:
            return

        clouds, self.buffer = self.buffer, []

        # Concatenate point payloads. All incoming clouds are assumed to share
        # the same field layout / point_step; we stack them as a single
        # height=1 unorganized cloud.
        latest = clouds[-1]
        data = bytearray()
        total_points = 0
        for c in clouds:
            data.extend(c.data)
            total_points += c.width * max(c.height, 1)

        source_frame = latest.header.frame_id
        out_frame = source_frame
        if self.target_frame and self.target_frame != source_frame:
            if self.ensure_transform(source_frame):
                self.transform_xyz_inplace(data, latest.point_step, latest.fields)
                out_frame = self.target_frame

        out = PointCloud2()
        out.header = latest.header
        out.header.frame_id = out_frame
        # Stamp with publish time, not the latest input cloud's stamp. self_filter
        # does per-body TF lookups with a 100ms timeout at the cloud's stamp; if
        # the stamp is older than the TF buffer's interpolation window the lookups
        # all time out and processing drops to ~0.4 Hz (26 bodies × 100ms).
        out.header.stamp = self.get_clock().now().to_msg()
        out.height = 1
        out.width = total_points
        out.fields = latest.fields
        out.is_bigendian = latest.is_bigendian
        out.point_step = latest.point_step
        out.row_step = latest.point_step * total_points
        out.data = bytes(data)
        out.is_dense = all(c.is_dense for c in clouds)

        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AccumulatePointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
