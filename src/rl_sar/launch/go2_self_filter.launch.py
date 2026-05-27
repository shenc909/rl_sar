# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Real-robot self-filter pipeline for the Go2 BEV lidar. Mirrors the self_filter
# node in gazebo.launch.py but feeds the *real* L1 cloud and uses wall-clock time.
#
#   real L1 cloud (cloud_in) -> robot_self_filter -> /lidar/points_filtered -> rl_real_go2_ros2
#
# robot_self_filter needs TF for every self_see_link (it masks points inside the
# robot's own body shapes). robot_state_publisher supplies the link tree from the
# URDF but needs JointState for the moving joints; the Go2 SDK only publishes
# /lowstate, so the llc_utils low_state_to_joint_states bridge (started here)
# republishes it as /joint_states.

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    cloud_in = LaunchConfiguration("cloud_in")

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

    # /lowstate -> /joint_states so robot_state_publisher can broadcast moving joint TF.
    low_state_to_joint_states_node = Node(
        package="llc_utils",
        executable="low_state_to_joint_states",
        name="low_state_to_joint_states",
        output="screen",
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
            ("/cloud_in", cloud_in),
            ("/cloud_out", "/lidar/points_filtered"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "cloud_in",
            description="Raw L1 lidar PointCloud2 topic in the sensor frame",
            default_value="/utlidar/cloud",
        ),
        robot_state_publisher_node,
        low_state_to_joint_states_node,
        self_filter_node,
    ])
