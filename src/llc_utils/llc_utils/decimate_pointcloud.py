# Downsample an incoming PointCloud2 by keeping every `stride`-th point and
# republish it, thinning the point count for downstream consumers that don't
# need full lidar density. stride=1 keeps every point (no thinning).
#
# Optionally re-expresses the cloud in a target frame via a one-shot (static)
# TF lookup, and/or re-stamps the output with publish time. Both default off,
# so the node is a pure sensor-frame decimator unless configured otherwise.
# (With stride=1 + target_frame set, it becomes a pure re-expression pass.)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener, TransformException


class DecimatePointCloud(Node):
    def __init__(self):
        super().__init__("decimate_pointcloud")

        self.declare_parameter("input_topic", "/utlidar/cloud")
        self.declare_parameter("output_topic", "/utlidar/cloud_decimated")
        # Keep every `stride`-th point. stride=1 disables decimation, turning the
        # node into a pure re-expression pass (e.g. sim: transform to base frame
        # without thinning); stride>1 downsamples.
        self.declare_parameter("stride", 10)
        # Empty target_frame disables the transform — the cloud is republished
        # in whatever frame the driver stamped on it.
        self.declare_parameter("target_frame", "")
        # Stamp the output with publish time instead of the input stamp.
        # self_filter does per-body TF lookups with a 100ms timeout at the
        # cloud's stamp; an input stamp older than the TF buffer's interpolation
        # window times them all out and drops processing to ~0.4 Hz.
        self.declare_parameter("restamp", False)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.restamp = bool(self.get_parameter("restamp").value)

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

        target_info = f" -> '{self.target_frame}'" if self.target_frame else ""
        stride_info = (
            "(pure pass-through)" if self.stride == 1
            else f"(keeping every {self.stride}th point)"
        )
        self.get_logger().info(
            f"Decimating {input_topic} -> {output_topic}{target_info} {stride_info}"
        )

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

    def cloud_callback(self, msg: PointCloud2):
        # Reshape the flat payload into one row per point, then keep every
        # other row. Slicing a height=1 unorganized cloud this way drops half
        # the points regardless of field layout.
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        # bytearray (not bytes) so the optional transform can edit xyz in place.
        # stride=1 keeps every point (pure pass-through / re-expression).
        data = bytearray(arr[::self.stride].tobytes())
        kept = len(data) // msg.point_step

        source_frame = msg.header.frame_id
        out_frame = source_frame
        if self.target_frame and self.target_frame != source_frame:
            if self.ensure_transform(source_frame):
                self.transform_xyz_inplace(data, msg.point_step, msg.fields)
                out_frame = self.target_frame

        out = PointCloud2()
        out.header = msg.header
        out.header.frame_id = out_frame
        if self.restamp:
            out.header.stamp = self.get_clock().now().to_msg()
        out.height = 1
        out.width = kept
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.point_step * kept
        out.data = bytes(data)
        out.is_dense = msg.is_dense

        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DecimatePointCloud()
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
