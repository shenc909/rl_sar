// Downsamples a dense Hesai lidar PointCloud2 to the angular pattern the policy
// was trained on (HesaiJT128 band), instead of a blind every-other-point cut:
//
//   * channel (ring) selection: keep rings in [channel_range[0], channel_range[1))
//     stepping by channel_skip   (default 29..127 step 3 -> 33 channels).
//   * horizontal selection: within horizontal_fov_range, keep one point per
//     horizontal_res-degree azimuth bin per channel (the point nearest the bin
//     centre), reproducing the trained sensor's azimuth columns.
//
// All per-point work (azimuth, masks, bin assignment, per-(channel,bin) nearest
// selection) is vectorised with libtorch so it runs in parallel: on CUDA if a
// GPU is visible, otherwise on multithreaded CPU. Requires the Hesai 'ring'
// field; clouds without ring/x/y are passed through unchanged.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace
{
// PointField.datatype -> byte width.
int datatype_size(uint8_t datatype)
{
    switch (datatype)
    {
        case sensor_msgs::msg::PointField::INT8:
        case sensor_msgs::msg::PointField::UINT8:    return 1;
        case sensor_msgs::msg::PointField::INT16:
        case sensor_msgs::msg::PointField::UINT16:   return 2;
        case sensor_msgs::msg::PointField::INT32:
        case sensor_msgs::msg::PointField::UINT32:
        case sensor_msgs::msg::PointField::FLOAT32:  return 4;
        case sensor_msgs::msg::PointField::FLOAT64:  return 8;
        default:                                     return 0;
    }
}

const sensor_msgs::msg::PointField *
find_field(const sensor_msgs::msg::PointCloud2 & msg, const std::string & name)
{
    for (const auto & f : msg.fields)
        if (f.name == name) return &f;
    return nullptr;
}
}  // namespace

class PointcloudDecimator : public rclcpp::Node
{
private:
    std::string pointcloud_in_topic_;
    std::string pointcloud_out_topic_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_sub_;

    // Downsample pattern.
    double hfov_min_, hfov_max_, hres_;
    int64_t ch_lo_, ch_hi_, ch_skip_;
    int64_t num_bins_;
    bool full_circle_;

    torch::Device device_;
    bool warned_fields_ = false;
    bool warned_bigendian_ = false;

public:
    PointcloudDecimator()
    : Node("pointcloud_decimator"),
      device_(torch::kCPU)
    {
        pointcloud_in_topic_ = declare_parameter<std::string>("pointcloud_in_topic", "/hesai/points");
        pointcloud_out_topic_ = declare_parameter<std::string>("pointcloud_out_topic", "/hesai/points_decimated");

        auto fov = declare_parameter<std::vector<double>>("horizontal_fov_range", {-180.0, 180.0});
        hres_ = declare_parameter<double>("horizontal_res", 4.0);
        auto chan = declare_parameter<std::vector<int64_t>>("channel_range", {29, 128});
        ch_skip_ = std::max<int64_t>(1, declare_parameter<int64_t>("channel_skip", 3));

        if (fov.size() < 2 || hres_ <= 0.0 || chan.size() < 2)
            throw std::runtime_error("Invalid horizontal_fov_range/horizontal_res/channel_range");

        hfov_min_ = fov[0];
        hfov_max_ = fov[1];
        ch_lo_ = chan[0];
        ch_hi_ = chan[1];

        num_bins_ = std::max<int64_t>(1, std::llround((hfov_max_ - hfov_min_) / hres_));
        full_circle_ = std::abs((hfov_max_ - hfov_min_) - 360.0) < 1e-3;

        if (torch::cuda::is_available())
            device_ = torch::Device(torch::kCUDA);

        auto qos = rclcpp::SensorDataQoS().keep_last(1).best_effort();
        pcl_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(pointcloud_out_topic_, qos);
        pcl_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_in_topic_, qos,
            [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { this->on_cloud(*msg); });

        RCLCPP_INFO(get_logger(),
            "Pointcloud decimator started on %s | %s -> %s | channels [%ld,%ld) step %ld, "
            "azimuth [%.1f,%.1f) @ %.1f deg (%ld bins)",
            device_.is_cuda() ? "CUDA" : "CPU",
            pointcloud_in_topic_.c_str(), pointcloud_out_topic_.c_str(),
            (long)ch_lo_, (long)ch_hi_, (long)ch_skip_,
            hfov_min_, hfov_max_, hres_, (long)num_bins_);
    }

private:
    // Reads the 'ring' channel column as an int64 tensor regardless of its
    // declared integer width (uint16 is the Hesai default).
    torch::Tensor read_ring(const torch::Tensor & bytes, int offset, uint8_t datatype)
    {
        const int w = datatype_size(datatype);
        auto col = bytes.slice(1, offset, offset + w).contiguous();
        switch (datatype)
        {
            case sensor_msgs::msg::PointField::UINT8:
            case sensor_msgs::msg::PointField::INT8:
                return col.squeeze(1).to(torch::kLong);
            case sensor_msgs::msg::PointField::UINT16:
            case sensor_msgs::msg::PointField::INT16:
                return col.view(torch::kInt16).squeeze(1).to(torch::kLong);
            case sensor_msgs::msg::PointField::UINT32:
            case sensor_msgs::msg::PointField::INT32:
                return col.view(torch::kInt32).squeeze(1).to(torch::kLong);
            default:
                return {};
        }
    }

    // Returns the selected point indices (Long, ascending) into the input cloud,
    // or an undefined tensor if the cloud cannot be pattern-downsampled.
    torch::Tensor select_indices(const sensor_msgs::msg::PointCloud2 & msg, int64_t n)
    {
        const auto * fx = find_field(msg, "x");
        const auto * fy = find_field(msg, "y");
        const auto * fr = find_field(msg, "ring");
        if (!fx || !fy || !fr) return {};
        if (fx->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
            fy->datatype != sensor_msgs::msg::PointField::FLOAT32)
            return {};

        auto raw = torch::from_blob(
            const_cast<uint8_t *>(msg.data.data()),
            {n, static_cast<int64_t>(msg.point_step)},
            torch::TensorOptions().dtype(torch::kUInt8));
        auto bytes = raw.to(device_);

        auto x = bytes.slice(1, fx->offset, fx->offset + 4).contiguous().view(torch::kFloat32).squeeze(1);
        auto y = bytes.slice(1, fy->offset, fy->offset + 4).contiguous().view(torch::kFloat32).squeeze(1);
        auto ring = read_ring(bytes, fr->offset, fr->datatype);
        if (!ring.defined()) return {};

        // Channel mask: ring in [ch_lo, ch_hi) and on the channel_skip lattice.
        auto chan_mask = ring.ge(ch_lo_)
                             .logical_and(ring.lt(ch_hi_))
                             .logical_and((ring - ch_lo_).remainder(ch_skip_).eq(0));

        auto az = torch::atan2(y, x) * (180.0 / M_PI);  // degrees in [-180, 180]
        torch::Tensor fov_mask = full_circle_
            ? torch::ones({n}, torch::TensorOptions().dtype(torch::kBool).device(device_))
            : az.ge(hfov_min_).logical_and(az.le(hfov_max_));

        auto valid_idx = chan_mask.logical_and(fov_mask).nonzero().squeeze(1);
        const int64_t m = valid_idx.size(0);
        if (m == 0) return valid_idx;

        auto az_v = az.index_select(0, valid_idx);
        auto ring_v = ring.index_select(0, valid_idx);

        auto bin = torch::floor((az_v - hfov_min_) / hres_ + 0.5).to(torch::kLong);
        bin = full_circle_ ? bin.remainder(num_bins_) : bin.clamp(0, num_bins_ - 1);

        // Wrapped angular distance to the bin centre (handles the -180/180 seam).
        auto centre = bin.to(torch::kFloat32) * static_cast<float>(hres_) + static_cast<float>(hfov_min_);
        auto diff = az_v - centre;
        auto err = (diff - 360.0 * torch::round(diff / 360.0)).abs();

        // One point per (channel, bin): sort by key, then err; keep first of each
        // key run. err < 180 << 1000 so the composite score is monotone in key.
        auto key = ring_v * num_bins_ + bin;
        auto score = key.to(torch::kFloat64) * 1000.0 + err.to(torch::kFloat64);
        auto order = std::get<1>(score.sort(0));
        auto sk = key.index_select(0, order);
        torch::Tensor first;
        if (m == 1)
        {
            first = torch::ones({1}, torch::TensorOptions().dtype(torch::kBool).device(device_));
        }
        else
        {
            auto head = torch::ones({1}, torch::TensorOptions().dtype(torch::kBool).device(device_));
            auto neq = sk.slice(0, 1, m).ne(sk.slice(0, 0, m - 1));
            first = torch::cat({head, neq}, 0);
        }
        auto chosen = valid_idx.index_select(0, order.masked_select(first));
        return std::get<0>(chosen.sort(0));
    }

    void on_cloud(const sensor_msgs::msg::PointCloud2 & msg)
    {
        torch::InferenceMode guard;

        if (msg.point_step == 0 || msg.data.empty())
        {
            pcl_pub_->publish(msg);
            return;
        }
        if (msg.is_bigendian && !warned_bigendian_)
        {
            RCLCPP_WARN(get_logger(), "Big-endian cloud not supported; passing through unchanged.");
            warned_bigendian_ = true;
        }

        const int64_t n = static_cast<int64_t>(msg.data.size() / msg.point_step);

        torch::Tensor chosen;
        if (!msg.is_bigendian)
        {
            try
            {
                chosen = select_indices(msg, n);
            }
            catch (const std::exception & e)
            {
                RCLCPP_ERROR(get_logger(), "Downsample failed (%s); passing cloud through.", e.what());
            }
        }

        if (!chosen.defined())
        {
            if (!warned_fields_)
            {
                RCLCPP_WARN(get_logger(),
                    "Cloud lacks usable x/y/ring fields; cannot pattern-downsample, passing through.");
                warned_fields_ = true;
            }
            pcl_pub_->publish(msg);
            return;
        }

        const int64_t kept = chosen.size(0);

        // Gather the selected point rows as a contiguous [kept, point_step] block.
        auto bytes = torch::from_blob(
            const_cast<uint8_t *>(msg.data.data()),
            {n, static_cast<int64_t>(msg.point_step)},
            torch::TensorOptions().dtype(torch::kUInt8)).to(device_);
        auto sel = bytes.index_select(0, chosen).to(torch::kCPU).contiguous();

        sensor_msgs::msg::PointCloud2 out;
        out.header = msg.header;
        out.height = 1;
        out.width = static_cast<uint32_t>(kept);
        out.fields = msg.fields;
        out.is_bigendian = msg.is_bigendian;
        out.point_step = msg.point_step;
        out.row_step = msg.point_step * static_cast<uint32_t>(kept);
        out.is_dense = msg.is_dense;
        out.data.resize(static_cast<size_t>(kept) * msg.point_step);
        if (kept > 0)
            std::memcpy(out.data.data(), sel.data_ptr<uint8_t>(), out.data.size());

        pcl_pub_->publish(out);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointcloudDecimator>();
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
