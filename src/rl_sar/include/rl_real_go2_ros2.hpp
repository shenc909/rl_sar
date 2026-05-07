/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 * Modified by Shen Chen for Go2 control via ROS2 topics
 */

#ifndef RL_REAL_GO2_ROS2_HPP
#define RL_REAL_GO2_ROS2_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_go2.hpp"
#include "fsm_go2w.hpp"

#include <csignal>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/wireless_controller.hpp"
#include "go2/motion_switch_client.hpp"

#include "robot_msgs/msg/fsm_state.hpp"
#include "robot_msgs/srv/set_fsm_state.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace unitree::robot::b2;
#define TOPIC_LOWCMD "/lowcmd"
#define TOPIC_LOWSTATE "/lowstate"
#define TOPIC_JOYSTICK "/wirelesscontroller"
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// union for joystick keys
typedef union
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} xKeySwitchUnion;

class RL_Real : public RL
{
public:
    RL_Real(int argc, char **argv);
    ~RL_Real();

    void ReturnToBuiltInLLCMode();
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

    // unitree interface
    void InitLowCmd();
    int QueryMotionStatus();
    std::string QueryServiceName(std::string form, std::string name);
    uint32_t Crc32Core(uint32_t *ptr, uint32_t len);
    void LowStateMessageHandler(const unitree_go::msg::LowState::SharedPtr message);
    void JoystickHandler(const unitree_go::msg::WirelessController::SharedPtr message);
    MotionSwitchClient msc;
    unitree_go::msg::LowCmd unitree_low_command{};
    unitree_go::msg::LowState unitree_low_state{};
    unitree_go::msg::WirelessController joystick{};

    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_publisher;
    unitree_go::msg::LowCmd low_cmd_msg{};
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscriber;
    unitree_go::msg::LowState low_state_msg{};
    rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr joystick_subscriber;
    unitree_go::msg::WirelessController joystick_msg{};
    xKeySwitchUnion unitree_joy;

    // others
    std::vector<float> mapped_joint_positions;
    std::vector<float> mapped_joint_velocities;
    double last_cmd_vel_time = 0.0;
    double last_navigation_toggle_time = 0.0;
    const double navigation_toggle_debounce_sec = 0.5;
    double last_height_scan_time = 0.0;

    std::vector<float> height_scan_obs;

    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    std_msgs::msg::Float32MultiArray height_scan;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr height_scan_subscriber;
    void HeightScanCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    rclcpp::Service<robot_msgs::srv::SetFsmState>::SharedPtr fsm_set_state_service;
    rclcpp::Publisher<robot_msgs::msg::FsmState>::SharedPtr fsm_state_publisher;
};

#endif // RL_REAL_GO2_ROS2_HPP
