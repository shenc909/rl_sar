#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

class HesaiRelay : public rclcpp::Node
{
    private:
        std::string pointcloud_in_topic_;
        std::string pointcloud_out_topic_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_sub_;
    public:
        HesaiRelay()
        : Node("hesai_relay")
        {
            pointcloud_in_topic_ = declare_parameter<std::string>("pointcloud_in_topic", "/lidar_points");
            pointcloud_out_topic_ = declare_parameter<std::string>("pointcloud_out_topic", "/hesai/points");
            auto qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();

            pcl_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(pointcloud_out_topic_, qos);
            pcl_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                pointcloud_in_topic_, qos,
                [this] (sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
                    pcl_pub_->publish(*msg);
                }
            );
            RCLCPP_INFO(this->get_logger(), "Relay started.");
        };
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HesaiRelay>();
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}