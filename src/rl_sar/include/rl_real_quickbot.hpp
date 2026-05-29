/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 * Modified by Shen Chen for Go2 control via standard ROS2 topics
 */

#ifndef RL_REAL_QUICKBOT_HPP
#define RL_REAL_QUICKBOT_HPP

// #define PLOT
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_quickbot.hpp"

#include <csignal>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joy.hpp>

#include "quickbot_interface/msg/motor_feedback.hpp"
#include "quickbot_interface/msg/motor_setpoints.hpp"
#include "robot_msgs/msg/fsm_state.hpp"
#include "robot_msgs/srv/set_fsm_state.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// Generic "ROS interface layer" Go2 node: consumes standard ROS2 topics
// (sensor_msgs Imu / Joy) plus quickbot_interface/MotorFeedback, and publishes
// joint targets as quickbot_interface/MotorSetpoints. No Unitree SDK.
#define TOPIC_JOINT_STATES "/bridge_node/motor_feedback"
#define TOPIC_IMU "/imu"
#define TOPIC_JOY "/bridge_node/joy"
#define TOPIC_CMD_VEL "/cmd_vel"
#define TOPIC_HEIGHT_SCAN "/local_elevation_array"
#define TOPIC_JOINT_COMMAND "/bridge_node/motor_setpoints"

class RL_Real : public RL
{
public:
    RL_Real(int argc, char **argv);
    ~RL_Real();

    std::shared_ptr<rclcpp::Node> ros2_node;

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void OnConfigSwitched() override;
    void RunModel();
    void RobotControl();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // joint-command publisher (quickbot_interface/MotorSetpoints, fixed-12 in bridge joint order)
    rclcpp::Publisher<quickbot_interface::msg::MotorSetpoints>::SharedPtr motor_setpoints_publisher;
    quickbot_interface::msg::MotorSetpoints motor_setpoints_msg;

    // standard sensor inputs
    rclcpp::Subscription<quickbot_interface::msg::MotorFeedback>::SharedPtr motor_feedback_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscriber;
    void MotorFeedbackCallback(const quickbot_interface::msg::MotorFeedback::SharedPtr msg);
    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void JoyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

    sensor_msgs::msg::Imu imu_msg;
    sensor_msgs::msg::Joy joy_msg;
    std::mutex state_mutex;
    quickbot_interface::msg::MotorFeedback latest_motor_feedback;
    bool joint_state_received = false;

    // others
    double last_cmd_vel_time = 0.0;
    double last_height_scan_time = 0.0;

    std::vector<float> height_scan_obs = std::vector<float>(187, 0.31f);

    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    std_msgs::msg::Float32MultiArray height_scan;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr height_scan_subscriber;
    void HeightScanCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    rclcpp::Service<robot_msgs::srv::SetFsmState>::SharedPtr fsm_set_state_service;
    rclcpp::Publisher<robot_msgs::msg::FsmState>::SharedPtr fsm_state_publisher;
};

#endif // RL_REAL_QUICKBOT_HPP
