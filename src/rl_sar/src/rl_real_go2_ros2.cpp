/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 * Modified by Shen Chen for Go2 control via ROS2 topics
 */

#include "rl_real_go2_ros2.hpp"

#include <atomic>
#include <csignal>
#include <termios.h>
#include <unistd.h>

namespace {
std::atomic<bool> g_sigint_requested{false};

void SigintHandler(int) {
    g_sigint_requested.store(true);
}
}  // namespace

RL_Real::RL_Real(int argc, char **argv)
    :   ros2_node(std::make_shared<rclcpp::Node>("rl_real_node")),
        msc(ros2_node.get())
{
    bool wheel_mode = (argc > 2 && std::string(argv[2]) == "wheel");

    // read params from yaml
    this->ang_vel_axis = "body";
    this->robot_name = wheel_mode ? "go2w" : "go2";
    this->ReadYaml(this->robot_name, "base.yaml");

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
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->height_scan_obs = std::vector<float>(this->params.Get<int>("num_height_scan_points"), 0.31f);
    this->InitLowCmd();
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // lowcmd publisher
    this->lowcmd_publisher = ros2_node->create_publisher<unitree_go::msg::LowCmd>(TOPIC_LOWCMD, rclcpp::SystemDefaultsQoS());

    // lowstate subscriber
    this->lowstate_subscriber = ros2_node->create_subscription<unitree_go::msg::LowState>(
        TOPIC_LOWSTATE, rclcpp::SystemDefaultsQoS(),
        std::bind(&RL_Real::LowStateMessageHandler, this, std::placeholders::_1)
    );

    // joystick subscriber
    this->joystick_subscriber = ros2_node->create_subscription<unitree_go::msg::WirelessController>(
        TOPIC_JOYSTICK, rclcpp::SystemDefaultsQoS(),
        std::bind(&RL_Real::JoystickHandler, this, std::placeholders::_1)
    );

    // cmd_vel subscriber
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        "/go2_1/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );

    // height_scan subscriber
    this->height_scan_subscriber = ros2_node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/go2_1/local_elevation_array", rclcpp::SystemDefaultsQoS(),
        [this] (const std_msgs::msg::Float32MultiArray::SharedPtr msg) {this->HeightScanCallback(msg);}
    );
    

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
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
}

RL_Real::~RL_Real()
{
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

    // KeyboardInterface() left stdin in non-canonical mode with ECHO off
    // (see kbhit() in rl_sdk.cpp). std::atexit only restores it after main
    // returns, so getline below would see empty/non-blocking reads. Re-enable
    // canonical line input and echo for the prompt.
    termios prompt_term;
    bool term_saved = (tcgetattr(STDIN_FILENO, &prompt_term) == 0);
    if (term_saved)
    {
        termios canonical = prompt_term;
        canonical.c_lflag |= (ICANON | ECHO);
        canonical.c_cc[VMIN] = 1;
        canonical.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &canonical);
    }

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

void RL_Real::GetState(RobotState<float> *state)
{
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
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        state->motor_state.q[i] = this->unitree_low_state.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].q;
        state->motor_state.dq[i] = this->unitree_low_state.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq;
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau_est;
    }
}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    unitree_go::msg::LowCmd dds_low_command{};
    dds_low_command.head[0] = 0xFE;
    dds_low_command.head[1] = 0xEF;
    dds_low_command.level_flag = 0xFF;
    dds_low_command.gpio = 0;

    for (int i = 0; i < 20; ++i)
    {
        dds_low_command.motor_cmd[i].mode = 0x01;
        dds_low_command.motor_cmd[i].q = PosStopF;
        dds_low_command.motor_cmd[i].kp = 0;
        dds_low_command.motor_cmd[i].dq = VelStopF;
        dds_low_command.motor_cmd[i].kd = 0;
        dds_low_command.motor_cmd[i].tau = 0;
    }

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        dds_low_command.motor_cmd[this->params.Get<std::vector<int>>("joint_mapping")[i]].mode = 0x01;
        dds_low_command.motor_cmd[this->params.Get<std::vector<int>>("joint_mapping")[i]].q = command->motor_command.q[i];
        dds_low_command.motor_cmd[this->params.Get<std::vector<int>>("joint_mapping")[i]].dq = command->motor_command.dq[i];
        dds_low_command.motor_cmd[this->params.Get<std::vector<int>>("joint_mapping")[i]].kp = command->motor_command.kp[i];
        dds_low_command.motor_cmd[this->params.Get<std::vector<int>>("joint_mapping")[i]].kd = command->motor_command.kd[i];
        dds_low_command.motor_cmd[this->params.Get<std::vector<int>>("joint_mapping")[i]].tau = command->motor_command.tau[i];
    }

    dds_low_command.crc = Crc32Core((uint32_t *)&dds_low_command, (sizeof(unitree_go::msg::LowCmd) >> 2) - 1);
    lowcmd_publisher->publish(dds_low_command);

#ifdef PLOT
    this->unitree_low_command = dds_low_command;
#endif
}

void RL_Real::RobotControl()
{
    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};

        if (this->control.navigation_mode)
        {
            this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};

            double cmd_vel_time_diff = ros2_node->now().seconds() - this->last_cmd_vel_time;
            if (cmd_vel_time_diff > 0.2)
            {   
                this->obs.commands = {0.0f, 0.0f, 0.0f};
                RCLCPP_WARN_THROTTLE(ros2_node->get_logger(), *ros2_node->get_clock(), 5000, "cmd_vel data stale, setting to zero!");
            }

        }

        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.height_scan = this->height_scan_obs;
        double height_scan_time_diff = ros2_node->now().seconds() - this->last_height_scan_time;
        if (height_scan_time_diff > 0.2) {
            RCLCPP_WARN_THROTTLE(ros2_node->get_logger(), *ros2_node->get_clock(), 1000, "Height map data stale, rate < 5Hz! Staleness: %f s", height_scan_time_diff);
        }

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est = this->robot_state.motor_state.tau_est;
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Real::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (!this->params.Get<std::vector<int>>("observations_history").empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        return clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
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
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state[i].q);
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd[i].q);
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
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

void RL_Real::LowStateMessageHandler(
    const unitree_go::msg::LowState::SharedPtr message
)
{
    this->unitree_low_state = *message;
}

void RL_Real::JoystickHandler(
    const unitree_go::msg::WirelessController::SharedPtr message
)
{
    joystick = *message;
    this->unitree_joy.value = joystick.keys;
}

void RL_Real::CmdvelCallback(
    const geometry_msgs::msg::Twist::SharedPtr msg
)
{
    this->cmd_vel = *msg;
    this->last_cmd_vel_time = ros2_node->now().seconds();
}

void RL_Real::HeightScanCallback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg
)
{
    this->height_scan = *msg;
    for (size_t i = 0; i < msg->data.size(); ++i) {
        RCLCPP_INFO(ros2_node->get_logger(), "  Data[%zu]: %f", i, msg->data[i]);
        // RCLCPP_INFO(this->get_logger(), "HM Received! Timestamp: %f", this->now().seconds());
        this->height_scan_obs[i] = msg->data[i];
    }
    this->last_height_scan_time = ros2_node->now().seconds();
}

int main(int argc, char **argv)
{
    if (argc < 1)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " [wheel]" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    
    // Keep ROS context alive through node destruction so destructor-time
    // service calls (e.g. SelectMode("mcf")) can still execute on Ctrl-C.
    // Use a plain init call for broad distro compatibility, then override
    // SIGINT handling to stop spinning before calling rclcpp::shutdown().
    rclcpp::init(argc, argv);
    std::signal(SIGINT, SigintHandler);

    auto rl_sar = std::make_shared<RL_Real>(argc, argv);
    while (rclcpp::ok() && !g_sigint_requested.load())
    {
        rclcpp::spin_some(rl_sar->ros2_node);
    }

    rl_sar->ReturnToBuiltInLLCMode();
    rl_sar.reset();

    rclcpp::shutdown();
    return 0;
}
