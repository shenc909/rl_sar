/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_REAL_GO2_ROS2_HPP
#define RL_REAL_GO2_ROS2_HPP

// #define PLOT
// #define CSV_LOGGER
// #define USE_ROS

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "loop.hpp"
#include "fsm.hpp"

// #include <unitree/idl/go2/LowState_.hpp>
// #include <unitree/idl/go2/LowCmd_.hpp>
// #include <unitree/idl/go2/WirelessController_.hpp>
// #include <unitree/common/time/time_tool.hpp>
// #include <unitree/common/thread/thread.hpp>
// #include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/wireless_controller.hpp"
#include "go2/motion_switch_client.hpp"
#include <csignal>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// using namespace unitree::common;
// using namespace unitree::robot;
using namespace unitree::robot::b2;
#define TOPIC_LOWCMD "/lowcmd"
#define TOPIC_LOWSTATE "/lowstate"
#define TOPIC_JOYSTICK "/wirelesscontroller"
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

// 遥控器键值联合体
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
    , public rclcpp::Node
{
public:
    RL_Real(bool wheel_mode);
    ~RL_Real();
    void ReturnToBuiltInLLCMode();

private:
    // rl functions
    torch::Tensor Forward() override;
    void GetState(RobotState<double> *state) override;
    void SetCommand(const RobotCommand<double> *command) override;
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
    std::vector<std::vector<double>> plot_real_joint_pos, plot_target_joint_pos;
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
    // unitree_go::msg::dds_::LowCmd_ unitree_low_command{};
    // unitree_go::msg::dds_::LowState_ unitree_low_state{};
    // unitree_go::msg::dds_::WirelessController_ joystick{};
    // ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    // ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    // ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_publisher;
    unitree_go::msg::LowCmd low_cmd_msg{};
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscriber;
    unitree_go::msg::LowState low_state_msg{};
    rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr joystick_subscriber;
    unitree_go::msg::WirelessController joystick_msg{};
    xKeySwitchUnion unitree_joy;

    // others
    int motiontime = 0;
    std::vector<double> mapped_joint_positions;
    std::vector<double> mapped_joint_velocities;

    std::vector<float> height_scan_obs = std::vector<float>(187, 0.31f);

    geometry_msgs::msg::Twist cmd_vel;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber;
    void CmdvelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    double last_cmd_vel_time = 0.0;
    double last_navigation_toggle_time = 0.0;
    const double navigation_toggle_debounce_sec = 0.5;
    std_msgs::msg::Float32MultiArray height_scan;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr height_scan_subscriber;
    void HeightScanCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    double last_height_scan_time = 0.0;
};

#endif // RL_REAL_GO2_ROS2_HPP
