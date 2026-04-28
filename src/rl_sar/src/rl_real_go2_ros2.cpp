/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_go2_ros2.hpp"

#include <atomic>
#include <csignal>

#include <rclcpp/executors/single_threaded_executor.hpp>

namespace {
std::atomic<bool> g_sigint_requested{false};

void SigintHandler(int) {
    g_sigint_requested.store(true);
}
}  // namespace

RL_Real::RL_Real(bool wheel_mode)
    : rclcpp::Node("rl_real_node"), msc(this)
{
    // read params from yaml
    this->ang_vel_type = "ang_vel_body";
    this->robot_name = wheel_mode ? "go2w" : "go2";
    this->ReadYamlBase(this->robot_name);

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init torch
    torch::autograd::GradMode::set_enabled(false);
    torch::set_num_threads(4);

    // init robot
    this->InitLowCmd();
    this->InitOutputs();
    this->InitControl();

    // lowstate subscriber
    this->lowstate_subscriber = this->create_subscription<unitree_go::msg::LowState>(
        TOPIC_LOWSTATE, rclcpp::SystemDefaultsQoS(),
        std::bind(&RL_Real::LowStateMessageHandler, this, std::placeholders::_1)
    );

    // joystick subscriber
    this->joystick_subscriber = this->create_subscription<unitree_go::msg::WirelessController>(
        TOPIC_JOYSTICK, rclcpp::SystemDefaultsQoS(),
        std::bind(&RL_Real::JoystickHandler, this, std::placeholders::_1)
    );

    // lowcmd publisher
    this->lowcmd_publisher = this->create_publisher<unitree_go::msg::LowCmd>(TOPIC_LOWCMD, rclcpp::SystemDefaultsQoS());

    // cmd_vel subscriber
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "/go2_1/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );

    // height_scan subscriber
    this->height_scan_subscriber = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/go2_1/local_elevation_array", rclcpp::SystemDefaultsQoS(),
        [this] (const std_msgs::msg::Float32MultiArray::SharedPtr msg) {this->HeightScanCallback(msg);}
    );

    // init MotionSwitcherClient
    // this->msc.SetTimeout(10.0f);
    // this->msc.Init();

    // Shut down motion control-related service
    while(this->QueryMotionStatus())
    {
        std::cout << "Try to deactivate the motion control-related service." << std::endl;
        int32_t ret = this->msc.ReleaseMode();
        if (ret == 0)
        {
            std::cout << "ReleaseMode succeeded." << std::endl;
        }
        else
        {
            std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
            throw std::runtime_error("Failed to release motion control-related service. Please rerun again.");
        }
        sleep(1);
    }

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
    // kill RL and control loops first
    // this->loop_rl->shutdown();
    // this->loop_control->shutdown();
    // this->loop_keyboard->shutdown();
    
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::ReturnToBuiltInLLCMode()
{
    this->loop_rl->shutdown();
    this->loop_control->shutdown();
    this->loop_keyboard->shutdown();
    bool return_to_llc = true;
    std::string user_input;
    std::cout << "Return to built-in LLC mode (mcf) before exit? [Y/n]: " << std::flush;
    if (std::getline(std::cin, user_input))
    {
        const auto first_non_ws = user_input.find_first_not_of(" \t\r\n");
        if (first_non_ws != std::string::npos)
        {
            const char answer = user_input[first_non_ws];
            if (answer == 'n' || answer == 'N')
            {
                return_to_llc = false;
            }
        }
    }

    if (return_to_llc)
    {
        const int32_t ret = this->msc.SelectMode("mcf");
        if (ret == 0)
        {
            std::cout << "SelectMode(\"mcf\") succeeded." << std::endl;
        }
        else
        {
            std::cout << "SelectMode(\"mcf\") failed. Error code: " << ret << std::endl;
        }
    }
    else
    {
        std::cout << "Skip returning to built-in LLC mode." << std::endl;
    }
}

void RL_Real::GetState(RobotState<double> *state)
{
    this->control.SetGamepad(Input::Gamepad::None); // reset gamepad state
    if (this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::A);
    if (this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::B);
    if (this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::X);
    if (this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->unitree_joy.components.L1) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->unitree_joy.components.R1) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.A) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.B) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.X) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.Y) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.F1) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.F2) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.up) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.down) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.left) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->unitree_joy.components.R1 && this->unitree_joy.components.right) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->unitree_joy.components.L1 && this->unitree_joy.components.R1) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->joystick.ly;
    this->control.y = -this->joystick.lx;
    this->control.yaw = -this->joystick.rx;

    state->imu.quaternion[0] = this->unitree_low_state.imu_state.quaternion[0]; // w
    state->imu.quaternion[1] = this->unitree_low_state.imu_state.quaternion[1]; // x
    state->imu.quaternion[2] = this->unitree_low_state.imu_state.quaternion[2]; // y
    state->imu.quaternion[3] = this->unitree_low_state.imu_state.quaternion[3]; // z

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->unitree_low_state.imu_state.gyroscope[i];
    }
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->unitree_low_state.motor_state[this->params.joint_mapping[i]].q;
        state->motor_state.dq[i] = this->unitree_low_state.motor_state[this->params.joint_mapping[i]].dq;
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state[this->params.joint_mapping[i]].tau_est;
    }
}

void RL_Real::SetCommand(const RobotCommand<double> *command)
{
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->unitree_low_command.motor_cmd[this->params.joint_mapping[i]].mode = 0x01;
        this->unitree_low_command.motor_cmd[this->params.joint_mapping[i]].q = command->motor_command.q[i];
        this->unitree_low_command.motor_cmd[this->params.joint_mapping[i]].dq = command->motor_command.dq[i];
        this->unitree_low_command.motor_cmd[this->params.joint_mapping[i]].kp = command->motor_command.kp[i];
        this->unitree_low_command.motor_cmd[this->params.joint_mapping[i]].kd = command->motor_command.kd[i];
        this->unitree_low_command.motor_cmd[this->params.joint_mapping[i]].tau = command->motor_command.tau[i];
    }

    this->unitree_low_command.crc = Crc32Core((uint32_t *)&unitree_low_command, (sizeof(unitree_go::msg::LowCmd) >> 2) - 1);
    lowcmd_publisher->publish(unitree_low_command);
}

void RL_Real::RobotControl()
{
    this->motiontime++;
    const double now_sec = this->now().seconds();

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0;
        this->control.y = 0;
        this->control.yaw = 0;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if ((this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
        && (now_sec - this->last_navigation_toggle_time >= this->navigation_toggle_debounce_sec))
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        this->last_navigation_toggle_time = now_sec;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
        this->control.current_keyboard = this->control.last_keyboard;
        // this->control.current_gamepad = this->control.last_gamepad;
    }
    
    this->control.last_gamepad = this->control.current_gamepad; // update last gamepad state for debouncing
    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->SetCommand(&this->robot_command);
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        if (this->control.navigation_mode)
        {
            double cmd_x = this->cmd_vel.linear.x;
            double cmd_y = this->cmd_vel.linear.y;
            double cmd_yaw = this->cmd_vel.angular.z;
            double cmd_vel_time_diff = this->now().seconds() - this->last_cmd_vel_time;
            if (cmd_vel_time_diff > 0.2)
            {
                cmd_x = 0.0;
                cmd_y = 0.0;
                cmd_yaw = 0.0;
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "cmd_vel data stale, setting to zero!");
            }
            this->obs.commands = torch::tensor({{cmd_x, cmd_y, cmd_yaw}});
        }
        else
        {
            this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        }
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        this->obs.height_scan = torch::tensor(this->height_scan_obs).unsqueeze(0);
        double height_scan_time_diff = this->now().seconds() - this->last_height_scan_time;
        if (height_scan_time_diff > 0.2) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Height map data stale, rate < 5Hz! Staleness: %f s", height_scan_time_diff);
        }

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
        {
            // this->TorqueLimitViaDofPos(this->output_dof_pos, this->output_dof_vel, this->obs.dof_vel);
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (this->output_dof_tau.defined() && this->output_dof_tau.numel() > 0)
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).unsqueeze(0);
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

torch::Tensor RL_Real::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions;
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd[i].q);
        plt::subplot(this->params.num_of_dofs, 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

uint32_t RL_Real::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }

    return CRC32;
}

void RL_Real::InitLowCmd()
{
    this->unitree_low_command.head[0] = 0xFE;
    this->unitree_low_command.head[1] = 0xEF;
    this->unitree_low_command.level_flag = 0xFF;
    this->unitree_low_command.gpio = 0;

    for (int i = 0; i < 20; ++i)
    {
        this->unitree_low_command.motor_cmd[i].mode = (0x01); // motor switch to servo (PMSM) mode
        this->unitree_low_command.motor_cmd[i].q = (PosStopF);
        this->unitree_low_command.motor_cmd[i].kp = (0);
        this->unitree_low_command.motor_cmd[i].dq = (VelStopF);
        this->unitree_low_command.motor_cmd[i].kd = (0);
        this->unitree_low_command.motor_cmd[i].tau = (0);
    }
}

int RL_Real::QueryMotionStatus()
{
    std::string robotForm, motionName;
    int motionStatus;
    int32_t ret = this->msc.CheckMode(robotForm, motionName);
    if (ret == 0)
    {
        std::cout << "CheckMode succeeded." << std::endl;
    }
    else
    {
        std::cout << "CheckMode failed. Error code: " << ret << std::endl;
    }
    if (motionName.empty())
    {
        std::cout << "The motion control-related service is deactivated." << std::endl;
        motionStatus = 0;
    }
    else
    {
        std::string serviceName = QueryServiceName(robotForm, motionName);
        std::cout << "Service: " << serviceName << " is activate" << std::endl;
        motionStatus = 1;
    }
    return motionStatus;
}

std::string RL_Real::QueryServiceName(std::string form, std::string name)
{
    if (form == "0")
    {
        if (name == "normal" )   return "sport_mode";
        if (name == "ai" )       return "ai_sport";
        if (name == "advanced" ) return "advanced_sport";
        if (name == "mcf" ) return "rl_mode";
    }
    else
    {
        if (name == "ai-w" )     return "wheeled_sport(go2W)";
        if (name == "normal-w" ) return "wheeled_sport(b2W)";
    }
    return "";
}

void RL_Real::LowStateMessageHandler(const unitree_go::msg::LowState::SharedPtr message)
{
    this->unitree_low_state = *message;
}

void RL_Real::JoystickHandler(const unitree_go::msg::WirelessController::SharedPtr message)
{
    joystick = *message;
    this->unitree_joy.value = joystick.keys;
}

void RL_Real::CmdvelCallback(
    const geometry_msgs::msg::Twist::SharedPtr msg
)
{
    this->cmd_vel = *msg;
    this->last_cmd_vel_time = this->now().seconds();
}

void RL_Real::HeightScanCallback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg
)
{
    this->height_scan = *msg;
    for (size_t i = 0; i < msg->data.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "  Data[%zu]: %f", i, msg->data[i]);
        // RCLCPP_INFO(this->get_logger(), "HM Received! Timestamp: %f", this->now().seconds());
        this->height_scan_obs[i] = msg->data[i];
    }
    this->last_height_scan_time = this->now().seconds();
}

int main(int argc, char **argv)
{
    if (argc < 1)
    {
        std::cout << "Usage: " << argv[0] << " [wheel]" << std::endl;
        exit(-1);
    }

    // Keep ROS context alive through node destruction so destructor-time
    // service calls (e.g. SelectMode("mcf")) can still execute on Ctrl-C.
    // Use a plain init call for broad distro compatibility, then override
    // SIGINT handling to stop spinning before calling rclcpp::shutdown().
    rclcpp::init(argc, argv);
    std::signal(SIGINT, SigintHandler);

    auto node = std::make_shared<RL_Real>(argc > 1 && std::string(argv[1]) == "wheel");
    // rclcpp::executors::SingleThreadedExecutor executor;
    // executor.add_node(node);

    while (rclcpp::ok() && !g_sigint_requested.load())
    {
        // executor.spin_some(std::chrono::milliseconds(50));
        rclcpp::spin_some(node);
    }

    node->ReturnToBuiltInLLCMode();
    // executor.remove_node(node);
    node.reset();
    rclcpp::shutdown();
    return 0;
}
