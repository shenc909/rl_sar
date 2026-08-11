/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BEV_RASTERIZER_HPP
#define BEV_RASTERIZER_HPP

// Shared BEV (bird's-eye-view) lidar rasterizer used by both rl_sim (simulated
// L1 cloud) and rl_real_go2_ros2 (real L1 cloud). Rasterizes a self-filtered
// point cloud into the 2-channel height_scan the dreamwaq_bev_direct policy was
// trained on: a grid_h x grid_w grid in the yaw-aligned base frame, channel-major
// [clearance(per_channel), occupancy(per_channel)]. Empty/occluded cells stay (0, 0).
// ROS2-only (depends on sensor_msgs::PointCloud2).

#if defined(USE_ROS2)

#include "rl_sdk.hpp"
#include "vector_math.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Rasterize the self-filtered lidar cloud into the 2-channel BEV height_scan
// (mirrors AccumulatedLidarBEV._emit). Returns fallback_fill on an empty cloud or
// when the configured grid does not match num_height_scan_points / num_channels.
inline std::vector<float> RasterizeLidarBEV(
    const sensor_msgs::msg::PointCloud2 &cloud,
    const YamlParams &params,
    const std::vector<float> &base_quat,
    const std::vector<float> &fallback_fill)
{
    if (cloud.width * cloud.height == 0)
    {
        return fallback_fill; // no scan yet -> flat-ground fallback
    }

    const int num_points = params.Get<int>("num_height_scan_points");
    const int num_channels = std::max(1, params.Get<int>("height_scan_num_channels", 2));
    const auto x_range = params.Get<std::vector<float>>("bev_x_range");
    const auto y_range = params.Get<std::vector<float>>("bev_y_range");
    const float res = params.Get<float>("bev_resolution");
    const std::string height_stat = params.Get<std::string>("bev_height_stat");
    const auto mount_xyz = params.Get<std::vector<float>>("bev_lidar_mount_xyz");
    const auto mount_rpy = params.Get<std::vector<float>>("bev_lidar_mount_rpy");
    // Optional vertical crop [z_min, z_max], applied in the yaw-aligned base frame
    // (height relative to the gravity-leveled base). Unset -> no crop, so existing
    // configs (e.g. the forward-down L1 sim) are unchanged. Useful for a top-mounted
    // sensor that sees ceilings/overhangs: without it, "max" aggregation lets an
    // overhead return dominate a cell and masquerade as terrain height.
    const auto z_range = params.Get<std::vector<float>>("bev_z_range");
    const bool crop_z = (z_range.size() == 2);
    const float z_min = crop_z ? z_range[0] : 0.0f;
    const float z_max = crop_z ? z_range[1] : 0.0f;

    const float x_min = x_range[0], x_max = x_range[1];
    const float y_min = y_range[0], y_max = y_range[1];
    const int grid_h = static_cast<int>(std::round((x_max - x_min) / res)) + 1;
    const int grid_w = static_cast<int>(std::round((y_max - y_min) / res)) + 1;
    const int cells = grid_h * grid_w;
    const int per_channel = num_points / num_channels;
    if (cells != per_channel)
    {
        std::cout << LOGGER::WARNING << "BEV grid " << grid_h << "x" << grid_w << "=" << cells
                  << " != per-channel " << per_channel << "; falling back to constant fill." << std::endl;
        return fallback_fill;
    }

    // Precompute the single affine map p_yaw = A * p_radar + b, where
    // A = R(q_yaw)^T R(q_base) R_mount  and  b = R(q_yaw)^T R(q_base) t_mount.
    using Mat3 = std::array<float, 9>; // row-major
    auto matmul = [](const Mat3 &a, const Mat3 &b) {
        Mat3 c{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
        return c;
    };
    auto transpose = [](const Mat3 &a) {
        return Mat3{a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
    };
    // R_mount from rpy = Rz(yaw) Ry(pitch) Rx(roll)
    const float cr = std::cos(mount_rpy[0]), sr = std::sin(mount_rpy[0]);
    const float cp = std::cos(mount_rpy[1]), sp = std::sin(mount_rpy[1]);
    const float cy = std::cos(mount_rpy[2]), sy = std::sin(mount_rpy[2]);
    const Mat3 R_mount{
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr};

    const std::vector<float> q_base = base_quat;                    // [w,x,y,z]
    const std::vector<float> q_yaw = QuaternionYawOnly(q_base);
    const auto rb_vec = QuaternionToRotationMatrix(q_base);          // R(q_base), 9 row-major
    const auto ry_vec = QuaternionToRotationMatrix(q_yaw);           // R(q_yaw)
    Mat3 R_base{}, R_yaw{};
    std::copy(rb_vec.begin(), rb_vec.end(), R_base.begin());
    std::copy(ry_vec.begin(), ry_vec.end(), R_yaw.begin());
    const Mat3 Ryaw_T = transpose(R_yaw);
    const Mat3 YB = matmul(Ryaw_T, R_base);                         // R(q_yaw)^T R(q_base)
    const Mat3 A = matmul(YB, R_mount);
    const std::array<float, 3> b{
        YB[0] * mount_xyz[0] + YB[1] * mount_xyz[1] + YB[2] * mount_xyz[2],
        YB[3] * mount_xyz[0] + YB[4] * mount_xyz[1] + YB[5] * mount_xyz[2],
        YB[6] * mount_xyz[0] + YB[7] * mount_xyz[1] + YB[8] * mount_xyz[2]};

    const bool use_mean = (height_stat == "mean");
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    std::vector<float> zmax(cells, NEG_INF), zsum(cells, 0.0f);
    std::vector<int> zcount(cells, 0);

    sensor_msgs::PointCloud2ConstIterator<float> it_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(cloud, "z");
    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z)
    {
        const float px = *it_x, py = *it_y, pz = *it_z;
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) continue;
        const float yx = A[0] * px + A[1] * py + A[2] * pz + b[0];
        const float yy = A[3] * px + A[4] * py + A[5] * pz + b[1];
        const float yz = A[6] * px + A[7] * py + A[8] * pz + b[2];
        if (crop_z && (yz < z_min || yz > z_max)) continue; // drop out-of-band (e.g. overhead) points
        const int ix = static_cast<int>(std::floor((yx - x_min) / res));
        const int iy = static_cast<int>(std::floor((yy - y_min) / res));
        if (ix < 0 || ix >= grid_h || iy < 0 || iy >= grid_w) continue;
        const int cell = ix * grid_w + iy;
        if (yz > zmax[cell]) zmax[cell] = yz;
        zsum[cell] += yz;
        zcount[cell] += 1;
    }

    std::vector<float> out(num_points, 0.0f);
    for (int c = 0; c < cells; ++c)
    {
        if (zcount[c] == 0) continue; // empty cell -> (0, 0)
        const float zagg = use_mean ? (zsum[c] / static_cast<float>(zcount[c])) : zmax[c];
        out[c] = -zagg;          // clearance = base-above-terrain (channel 0)
        out[per_channel + c] = 1.0f; // occupancy (channel 1)
    }
    return out;
}

#endif // USE_ROS2

#endif // BEV_RASTERIZER_HPP
