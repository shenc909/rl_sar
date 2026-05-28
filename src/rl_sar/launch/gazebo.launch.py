# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rname = LaunchConfiguration("rname")
    gui = LaunchConfiguration("gui")
    use_joy = LaunchConfiguration("use_joy")

    wname = "stairs"
    robot_name = ParameterValue(Command(["echo -n ", rname]), value_type=str)
    ros_namespace = ParameterValue(Command(["echo -n ", "/", rname, "_gazebo"]), value_type=str)
    gazebo_model_name = ParameterValue(Command(["echo -n ", rname, "_gazebo"]), value_type=str)

    robot_description = ParameterValue(
        Command([
            "xacro ",
            Command(["echo -n ", Command(["ros2 pkg prefix ", rname, "_description"])]),
            "/share/", rname, "_description/xacro/robot.xacro"
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            # "verbose": "true",
            # "pause": "true",  # Not Available
            "world": os.path.join(get_package_share_directory("rl_sar"), "worlds", wname + ".world"),
            # gui:=false runs gzserver headless. Useful when gzclient hangs at startup
            # with a game controller plugged in (a Gazebo-Classic GUI + joystick issue).
            "gui": gui,
        }.items(),
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "/robot_description",
            "-entity", "robot_model",
            "-z", "1.0",
        ],
        output="screen",
    )

    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable='spawner.py' if os.environ.get('ROS_DISTRO', '') == 'foxy' else 'spawner',
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    robot_joint_controller_node = Node(
        package="controller_manager",
        executable='spawner.py' if os.environ.get('ROS_DISTRO', '') == 'foxy' else 'spawner',
        arguments=["robot_joint_controller"],
        output="screen",
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        condition=IfCondition(use_joy),
        # The joy package links SDL2. SDL's default init grabs real video/audio
        # drivers (X11/GL/ALSA), which races with the Gazebo-Classic GUI's own
        # GL/X11 startup and intermittently hangs gzclient (with or without a
        # controller attached). Forcing the dummy drivers keeps SDL's joystick
        # subsystem fully working while never touching the display gzclient owns,
        # which removes the race at its source.
        additional_env={'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'},
        parameters=[{
            'deadzone': 0.1,
            'autorepeat_rate': 0.0,
        }],
    )
    # Belt-and-suspenders: also start joy_node only after gzclient has come up, so
    # the two never initialize concurrently even if the env guard above is incomplete.
    delayed_joy_node = TimerAction(period=8.0, actions=[joy_node])

    param_node = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="param_node",
        parameters=[{
            "robot_name": robot_name,
            "gazebo_model_name": gazebo_model_name,
        }],
    )

    # robot_self_filter: strips the robot's own body from the BEV lidar cloud (/lidar/points ->
    # /lidar/points_filtered) so rl_sim only rasterizes terrain. go2-specific; disable with bev_lidar:=false
    # for robots without the lidar. The filter config (link names) lives in the go2_description package.
    self_filter_config = os.path.join(
        get_package_share_directory("go2_description"), "config", "self_filter.yaml"
    )
    self_filter_node = Node(
        package="robot_self_filter",
        executable="self_filter",
        name="self_filter",
        output="screen",
        condition=IfCondition(LaunchConfiguration("bev_lidar")),
        parameters=[
            self_filter_config,
            {
                "robot_description": robot_description,
                "lidar_sensor_type": 0,  # 0 = XYZSensor (plain Gazebo PointCloud2)
                "zero_for_removed_points": False,
                "use_sim_time": True,
            },
        ],
        remappings=[
            ("/cloud_in", "/lidar/points"),
            ("/cloud_out", "/lidar/points_filtered"),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "rname",
            description="Robot name (e.g., a1, go2)",
            default_value=TextSubstitution(text=""),
        ),
        DeclareLaunchArgument(
            "bev_lidar",
            description="Run the BEV lidar self-filter node (go2 only)",
            "gui",
            description="Launch the Gazebo client GUI (set false if gzclient hangs with a joystick attached)",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "use_joy",
            description="Start joy_node (set false to launch without grabbing the joystick device)",
            default_value="true",
        ),
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        joint_state_broadcaster_node,
        # robot_joint_controller_node,  # Spawn in rl_sim.cpp
        delayed_joy_node,
        param_node,
        self_filter_node,
    ])

