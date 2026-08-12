# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Real-robot TF pipeline for the Go2 dreamwaq_bev_direct policy.
#
# The BEV cloud comes from a top-mounted MID-360, already decimated and already in
# go2_1/base_link, so it is fed straight to rl_real (route it onto
# /lidar/points_filtered via a remap on the rl_real side). The MID-360 cannot see
# the robot's own body, so no self-filtering is needed. The old L1 flow
# (decimate_pointcloud -> robot_self_filter) is kept commented below for reference.
#
# What this launch still does: publish the robot's TF tree on /go2_1/tf so external
# tools (rviz, debugging) have it. The go2_1 driver publishes base_link there but
# not the leg links; robot_state_publisher supplies the URDF link tree and
# low_state_to_joint_states bridges the Go2 SDK's /lowstate to /joint_states so the
# moving joint TF gets broadcast. base_link_to_base_tf_node joins the driver's
# go2_1/base_link to the URDF root `base` (base_link is absent from the URDF).

import os
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
# Re-enable with the commented L1 decimate/self_filter flow below:
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Args for the disabled L1 decimate/self_filter flow; kept commented alongside
    # the nodes that used them (see below), re-enable together if switching back.
    # cloud_in = LaunchConfiguration("cloud_in")
    # decimated_topic = LaunchConfiguration("decimated_topic")
    # lidar_frame = LaunchConfiguration("lidar_frame")
    # lidar_parent_frame = LaunchConfiguration("lidar_parent_frame")
    # lidar_yaw = LaunchConfiguration("lidar_yaw")
    # lidar_pitch = LaunchConfiguration("lidar_pitch")
    # lidar_roll = LaunchConfiguration("lidar_roll")

    robot_description = ParameterValue(
        Command([
            "xacro ",
            os.path.join(
                get_package_share_directory("go2_description"), "xacro", "robot.xacro"
            ),
        ]),
        value_type=str,
    )

    # The go2_1 driver publishes TF on /go2_1/tf[_static], not the default /tf.
    # Remap every node that produces or consumes TF onto the namespaced topics so
    # the driver's base_link, this RSP's leg-link tree, and the static mounts all
    # share one coherent tree. Frame *ids* stay bare URDF names (base, FL_hip, ...)
    # because self_filter looks them up unprefixed; only the topic is namespaced.
    tf_remaps = [
        ("/tf", "/go2_1/tf"),
        ("/tf_static", "/go2_1/tf_static"),
    ]

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": False}],
        remappings=tf_remaps,
    )

    low_state_to_joint_states_node = Node(
        package="llc_utils",
        executable="low_state_to_joint_states",
        name="low_state_to_joint_states",
        output="screen",
    )

    # The driver's tree is rooted at go2_1/base_link; the Go2 URDF tree is rooted
    # at `base` (base_link does not exist in the URDF). Bolt them together with a
    # zero static transform so self_filter can resolve the cloud frame
    # (go2_1/base_link) against every self_see_link. Parent must be go2_1/base_link
    # and child `base`, since RSP is the only publisher of `base`.
    base_link_to_base_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_base",
        output="screen",
        arguments=[
            "0", "0", "0",
            "0", "0", "0",
            "go2_1/base_link",
            "base",
        ],
        remappings=tf_remaps,
    )

    # NOTE: disabled for this base_link-direct pipeline. The cloud now arrives
    # already in go2_1/base_link, so we no longer bridge the utlidar_lidar sensor
    # frame into the URDF tree here (that role moved to base_link_to_base_tf_node).
    # Kept commented rather than deleted because the utlidar_lidar frame still
    # physically exists; re-enable if switching back to a sensor-frame cloud.
    # utlidar_static_tf_node = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="radar_to_utlidar_lidar",
    #     output="screen",
    #     arguments=[
    #         "0", "0", "0",
    #         lidar_yaw, lidar_pitch, lidar_roll,
    #         lidar_parent_frame,
    #         lidar_frame,
    #     ],
    # )

    # NOTE: disabled for this pipeline. The cloud source is a top-mounted MID-360,
    # already decimated and already in go2_1/base_link, so the decimator is
    # unnecessary. Kept commented rather than deleted; re-enable if a raw/dense
    # cloud source needs thinning (and set target_frame if it also needs
    # re-expressing). The pattern params below downsample a dense Hesai cloud to
    # the HesaiJT128 band the policy trained on (channels 29..127 step 3, azimuth
    # [-180,180) at 4 deg); they are ignored for clouds without a 'ring' field.
    # decimate_pointcloud_node = Node(
    #     package="llc_utils",
    #     executable="decimate_pointcloud",
    #     name="decimate_pointcloud",
    #     output="screen",
    #     parameters=[{
    #         "input_topic": cloud_in,
    #         "output_topic": decimated_topic,
    #         "horizontal_fov_range": [-180.0, 180.0],
    #         "horizontal_res": 4.0,
    #         "channel_range": [29, 128],
    #         "channel_skip": 3,
    #         "target_frame": "",
    #         "restamp": True,
    #     }],
    # )

    # NOTE: disabled for this pipeline. The MID-360 is top-mounted and cannot see
    # the robot's own body, so there are no self-points to strip. rl_real subscribes
    # to /lidar/points_filtered directly (route the driver's cloud there via a remap
    # on the rl_real side). Kept commented rather than deleted; re-enable for a
    # body-visible sensor mount.
    # self_filter_config = os.path.join(
    #     get_package_share_directory("go2_description"), "config", "self_filter.yaml"
    # )
    # self_filter_node = Node(
    #     package="robot_self_filter",
    #     executable="self_filter",
    #     name="self_filter",
    #     output="screen",
    #     parameters=[
    #         self_filter_config,
    #         {
    #             "robot_description": robot_description,
    #             "lidar_sensor_type": 0,  # 0 = XYZSensor (plain PointCloud2)
    #             "zero_for_removed_points": False,
    #             "use_sim_time": False,
    #         },
    #     ],
    #     remappings=[
    #         ("/cloud_in", decimated_topic),
    #         ("/cloud_out", "/lidar/points_filtered"),
    #     ],
    # )

    return LaunchDescription([
        # Args for the disabled L1 decimate/self_filter flow; re-enable together
        # with the commented LaunchConfiguration bindings and nodes above.
        # DeclareLaunchArgument(
        #     "cloud_in",
        #     description="Raw L1 lidar PointCloud2 topic in the sensor frame",
        #     default_value="/lidar_points",
        # ),
        # DeclareLaunchArgument(
        #     "decimated_topic",
        #     description="Topic carrying the decimator output (fed into self_filter)",
        #     default_value="/lidar/points_decimated",
        # ),
        # DeclareLaunchArgument(
        #     "lidar_frame",
        #     description="frame_id stamped on the real L1 PointCloud2",
        #     default_value="go2_1/base_link",
        # ),
        # DeclareLaunchArgument(
        #     "lidar_parent_frame",
        #     description="URDF link the L1 sensor frame attaches to",
        #     default_value="go2_1/base_link",
        # ),
        # DeclareLaunchArgument(
        #     "lidar_yaw",
        #     description="L1 sensor yaw (rad) relative to radar link; default -45 deg",
        #     default_value="0.0",
        # ),
        # DeclareLaunchArgument(
        #     "lidar_pitch",
        #     description="L1 sensor pitch (rad) relative to radar link",
        #     default_value="0.0",
        # ),
        # DeclareLaunchArgument(
        #     "lidar_roll",
        #     description="L1 sensor roll (rad) relative to radar link; default -15 deg",
        #     default_value="0.0",
        # ),
        robot_state_publisher_node,
        low_state_to_joint_states_node,
        base_link_to_base_tf_node,
    ])
