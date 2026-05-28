# Accumulate incoming PointCloud2 messages from /utlidar/cloud and republish
# the concatenated cloud on a fixed cadence (default 10 Hz), resetting the
# buffer after each publish.

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2


class AccumulatePointCloud(Node):
    def __init__(self):
        super().__init__("accumulate_pointcloud")

        self.declare_parameter("input_topic", "/utlidar/cloud")
        self.declare_parameter("output_topic", "/utlidar/cloud_accumulated")
        self.declare_parameter("publish_rate_hz", 10.0)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.buffer = []

        self.publisher = self.create_publisher(PointCloud2, output_topic, 10)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.cloud_callback, qos_profile_sensor_data
        )
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.publish_accumulated)

        self.get_logger().info(
            f"Accumulating {input_topic} -> {output_topic} @ {publish_rate_hz:.1f} Hz"
        )

    def cloud_callback(self, msg: PointCloud2):
        self.buffer.append(msg)

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

        out = PointCloud2()
        out.header = latest.header
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
