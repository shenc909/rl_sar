# Downsample an incoming PointCloud2 and republish it. Two modes, picked per
# cloud by whether the Hesai 'ring' field is present:
#
#   * pattern mode (cloud has ring/x/y): reproduce the angular pattern the policy
#     was trained on, instead of a blind every-Nth-point cut.
#       - channel (ring) selection: keep rings in [channel_range[0], channel_range[1])
#         stepping by channel_skip  (default 29..127 step 3 -> 33 channels, the
#         HesaiJT128 band the quickbot policy trained on).
#       - horizontal selection: within horizontal_fov_range, keep one point per
#         horizontal_res-degree azimuth bin per channel (the point nearest the bin
#         centre), reproducing the trained sensor's azimuth columns.
#   * stride mode (no ring field, e.g. the Gazebo lidar): keep every `stride`-th
#     point. stride=1 keeps every point, so the node becomes a pure pass-through.
#
# Optionally re-expresses the cloud in a target frame via a one-shot (static) TF
# lookup, and/or re-stamps the output with publish time. Both default off, so the
# node is a pure sensor-frame downsampler unless configured otherwise. (With a
# non-ring cloud, stride=1 and target_frame set, it is a pure re-expression pass.)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener, TransformException


# PointField.datatype -> numpy scalar code.
_PF_TO_NP = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


class DecimatePointCloud(Node):
    def __init__(self):
        super().__init__("decimate_pointcloud")

        self.declare_parameter("input_topic", "/utlidar/cloud")
        self.declare_parameter("output_topic", "/utlidar/cloud_decimated")
        # Pattern mode: lidar downsample pattern (defaults match the trained
        # HesaiJT128 band). Used when the cloud carries a 'ring' field.
        self.declare_parameter("horizontal_fov_range", [-180.0, 180.0])
        self.declare_parameter("horizontal_res", 4.0)
        self.declare_parameter("channel_range", [29, 128])  # half-open [lo, hi)
        self.declare_parameter("channel_skip", 3)
        # Stride mode (no 'ring' field): keep every `stride`-th point. stride=1
        # disables thinning, turning the node into a pure re-expression pass
        # (e.g. sim: transform to base frame without dropping points).
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
        fov = list(self.get_parameter("horizontal_fov_range").value)
        self.hfov_min, self.hfov_max = float(fov[0]), float(fov[1])
        self.hres = float(self.get_parameter("horizontal_res").value)
        ch = list(self.get_parameter("channel_range").value)
        self.ch_lo, self.ch_hi = int(ch[0]), int(ch[1])
        self.ch_skip = max(1, int(self.get_parameter("channel_skip").value))
        self.target_frame = self.get_parameter("target_frame").value
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.restamp = bool(self.get_parameter("restamp").value)

        # Number of azimuth bins; wrap bins when the FOV spans the full circle so
        # the -180/180 seam collapses to a single column.
        self.num_bins = max(1, int(round((self.hfov_max - self.hfov_min) / self.hres)))
        self.full_circle = abs((self.hfov_max - self.hfov_min) - 360.0) < 1e-3

        # Cached source->target transform.
        self.cached_R = None
        self.cached_t = None
        self.cached_source = None
        self.tf_warned = False
        self.ring_warned = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(PointCloud2, output_topic, 10)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.cloud_callback, qos_profile_sensor_data
        )

        target_info = f" -> '{self.target_frame}'" if self.target_frame else ""
        stride_info = (
            "pure pass-through" if self.stride == 1
            else f"every {self.stride}th point"
        )
        self.get_logger().info(
            f"Downsampling {input_topic} -> {output_topic}{target_info} | "
            f"ring pattern: channels [{self.ch_lo},{self.ch_hi}) step {self.ch_skip}, "
            f"azimuth [{self.hfov_min},{self.hfov_max}) @ {self.hres} deg | "
            f"no-ring fallback: {stride_info}"
        )

    def select_indices(self, msg: PointCloud2):
        # Read just x, y, ring as a structured view over the raw payload (honours
        # arbitrary field offsets/padding via an explicit itemsize=point_step).
        by_name = {f.name: f for f in msg.fields}
        if "ring" not in by_name or "x" not in by_name or "y" not in by_name:
            return None  # not a Hesai-style cloud we can pattern-downsample
        endian = ">" if msg.is_bigendian else "<"
        names, formats, offsets = [], [], []
        for nm in ("x", "y", "ring"):
            f = by_name[nm]
            if f.datatype not in _PF_TO_NP:
                return None
            names.append(nm)
            formats.append(endian + _PF_TO_NP[f.datatype])
            offsets.append(f.offset)
        dt = np.dtype({
            "names": names, "formats": formats,
            "offsets": offsets, "itemsize": msg.point_step,
        })
        rec = np.frombuffer(msg.data, dtype=dt)
        if rec.size == 0:
            return np.empty(0, dtype=np.int64)

        ring = rec["ring"].astype(np.int64)
        x = rec["x"].astype(np.float64)
        y = rec["y"].astype(np.float64)

        # Channel mask: in [ch_lo, ch_hi) and on the channel_skip lattice.
        chan_mask = (
            (ring >= self.ch_lo)
            & (ring < self.ch_hi)
            & (((ring - self.ch_lo) % self.ch_skip) == 0)
        )

        az = np.degrees(np.arctan2(y, x))  # [-180, 180]
        if self.full_circle:
            fov_mask = np.ones(az.shape, dtype=bool)
        else:
            fov_mask = (az >= self.hfov_min) & (az <= self.hfov_max)

        valid = chan_mask & fov_mask
        valid_idx = np.nonzero(valid)[0]
        if valid_idx.size == 0:
            return valid_idx

        az_v = az[valid_idx]
        ring_v = ring[valid_idx]
        bin_raw = np.floor((az_v - self.hfov_min) / self.hres + 0.5).astype(np.int64)
        if self.full_circle:
            bin_v = np.mod(bin_raw, self.num_bins)
        else:
            bin_v = np.clip(bin_raw, 0, self.num_bins - 1)

        # Distance from each point to its bin centre, wrapped to [-180, 180).
        centre = self.hfov_min + bin_v * self.hres
        diff = (az_v - centre + 180.0) % 360.0 - 180.0
        err = np.abs(diff)

        # One point per (channel, bin): sort by key then err, keep first of each.
        key = ring_v * self.num_bins + bin_v
        order = np.lexsort((err, key))
        sk = key[order]
        first = np.empty(sk.shape, dtype=bool)
        first[0] = True
        np.not_equal(sk[1:], sk[:-1], out=first[1:])
        chosen_local = order[first]
        return np.sort(valid_idx[chosen_local])

    def ensure_transform(self, source_frame: str) -> bool:
        # The radar<->lidar TF is static, so a one-shot lookup is enough; cache
        # it and only re-lookup if the source frame changes (it shouldn't).
        if self.cached_R is not None and self.cached_source == source_frame:
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
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

        chosen = self.select_indices(msg)
        if chosen is None:
            # No ring field: fall back to blind stride thinning (stride=1 keeps
            # every point, i.e. a pure pass-through / re-expression).
            if not self.ring_warned:
                self.get_logger().warn(
                    "Cloud has no 'ring'/x/y fields; cannot pattern-downsample, "
                    f"falling back to stride={self.stride}."
                )
                self.ring_warned = True
            # bytearray (not bytes) so the optional transform can edit xyz in place.
            data = bytearray(arr[::self.stride].tobytes())
            kept = len(data) // msg.point_step
        else:
            kept = int(chosen.size)
            # Fancy-index the selected point rows; bytearray so the optional
            # transform can edit xyz in place.
            data = bytearray(arr[chosen].tobytes())

        source_frame = msg.header.frame_id
        out_frame = source_frame
        if self.target_frame and self.target_frame != source_frame and kept > 0:
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
