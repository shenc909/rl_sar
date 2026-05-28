# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Real-robot perception pipeline for the Go2 dreamwaq_bev_direct policy. Stacks
# an accumulator in front of robot_self_filter so the BEV rasterizer sees a
# denser cloud than a single L1 sweep provides:
#
#   real L1 cloud (cloud_in) -> accumulate_pointcloud -> /utlidar/cloud_accumulated
#                            -> robot_self_filter     -> /lidar/points_filtered
#                            -> rl_real_go2_ros2
#
# robot_self_filter needs TF for every self_see_link. robot_state_publisher
# supplies the link tree from the URDF; low_state_to_joint_states bridges the
# Go2 SDK's /lowstate to /joint_states so moving joint TF gets broadcast.

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    cloud_in = LaunchConfiguration("cloud_in")
    accumulated_topic = LaunchConfiguration("accumulated_topic")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    lidar_frame = LaunchConfiguration("lidar_frame")
    lidar_parent_frame = LaunchConfiguration("lidar_parent_frame")
    lidar_yaw = LaunchConfiguration("lidar_yaw")
    lidar_pitch = LaunchConfiguration("lidar_pitch")
    lidar_roll = LaunchConfiguration("lidar_roll")

    robot_description = ParameterValue(
        Command([
            "xacro ",
            os.path.join(
                get_package_share_directory("go2_description"), "xacro", "robot.xacro"
            ),
        ]),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": False}],
    )

    low_state_to_joint_states_node = Node(
        package="llc_utils",
        executable="low_state_to_joint_states",
        name="low_state_to_joint_states",
        output="screen",
    )

    # Real L1 cloud arrives in `utlidar_lidar`, which is not in the Go2 URDF; bolt
    # it to the existing `radar` link (the URDF's L1 mount) so robot_self_filter
    # can resolve the cloud's frame against the robot tree. The URDF's `radar`
    # joint orientation does not match the real L1 sensor axes — observed offset
    # is roughly -45 deg yaw and -15 deg roll (tune via launch args).
    # Positional args (x y z yaw pitch roll parent child) — the only form Foxy's
    # static_transform_publisher accepts; the --frame-id/--x flag syntax is
    # Humble+ only and silently no-ops here.
    utlidar_static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="radar_to_utlidar_lidar",
        output="screen",
        arguments=[
            "0", "0", "0",
            lidar_yaw, lidar_pitch, lidar_roll,
            lidar_parent_frame,
            lidar_frame,
        ],
    )

    accumulate_pointcloud_node = Node(
        package="llc_utils",
        executable="accumulate_pointcloud",
        name="accumulate_pointcloud",
        output="screen",
        parameters=[{
            "input_topic": cloud_in,
            "output_topic": accumulated_topic,
            "publish_rate_hz": publish_rate_hz,
        }],
    )

    self_filter_config = os.path.join(
        get_package_share_directory("go2_description"), "config", "self_filter.yaml"
    )
    self_filter_node = Node(
        package="robot_self_filter",
        executable="self_filter",
        name="self_filter",
        output="screen",
        parameters=[
            self_filter_config,
            {
                "robot_description": robot_description,
                "lidar_sensor_type": 0,  # 0 = XYZSensor (plain PointCloud2)
                "zero_for_removed_points": False,
                "use_sim_time": False,
            },
        ],
        remappings=[
            ("/cloud_in", accumulated_topic),
            ("/cloud_out", "/lidar/points_filtered"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "cloud_in",
            description="Raw L1 lidar PointCloud2 topic in the sensor frame",
            default_value="/utlidar/cloud",
        ),
        DeclareLaunchArgument(
            "accumulated_topic",
            description="Topic carrying the accumulator output (fed into self_filter)",
            default_value="/utlidar/cloud_accumulated",
        ),
        DeclareLaunchArgument(
            "publish_rate_hz",
            description="Accumulator republish rate; buffer is flushed each tick",
            default_value="10.0",
        ),
        DeclareLaunchArgument(
            "lidar_frame",
            description="frame_id stamped on the real L1 PointCloud2",
            default_value="utlidar_lidar",
        ),
        DeclareLaunchArgument(
            "lidar_parent_frame",
            description="URDF link the L1 sensor frame attaches to",
            default_value="radar",
        ),
        DeclareLaunchArgument(
            "lidar_yaw",
            description="L1 sensor yaw (rad) relative to radar link; default -45 deg",
            default_value="-0.7853981633974483",
        ),
        DeclareLaunchArgument(
            "lidar_pitch",
            description="L1 sensor pitch (rad) relative to radar link",
            default_value="0.0",
        ),
        DeclareLaunchArgument(
            "lidar_roll",
            description="L1 sensor roll (rad) relative to radar link; default -15 deg",
            default_value="-0.2617993877991494",
        ),
        robot_state_publisher_node,
        low_state_to_joint_states_node,
        utlidar_static_tf_node,
        accumulate_pointcloud_node,
        self_filter_node,
    ])
