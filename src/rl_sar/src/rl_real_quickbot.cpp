/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 * Modified by Shen Chen for Go2 control via standard ROS2 topics
 */

#include "rl_real_quickbot.hpp"

#include <atomic>
#include <csignal>

namespace {
std::atomic<bool> g_sigint_requested{false};

void SigintHandler(int) {
    g_sigint_requested.store(true);
}
}  // namespace

RL_Real::RL_Real(int argc, char **argv)
    : ros2_node(std::make_shared<rclcpp::Node>("rl_real_quickbot_node"))
{
    (void)argc;
    (void)argv;

    // read params from yaml
    this->ang_vel_axis = "body";
    this->robot_name = "quickbot";
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
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // joint-command publisher (quickbot_interface/MotorSetpoints, fixed-12 bridge order)
    this->motor_setpoints_publisher = ros2_node->create_publisher<quickbot_interface::msg::MotorSetpoints>(
        TOPIC_JOINT_COMMAND, rclcpp::SystemDefaultsQoS());

    // standard sensor subscribers
    this->motor_feedback_subscriber = ros2_node->create_subscription<quickbot_interface::msg::MotorFeedback>(
        TOPIC_JOINT_STATES, rclcpp::SensorDataQoS(),
        [this] (const quickbot_interface::msg::MotorFeedback::SharedPtr msg) {this->MotorFeedbackCallback(msg);}
    );
    this->imu_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Imu>(
        TOPIC_IMU, rclcpp::SensorDataQoS(),
        [this] (const sensor_msgs::msg::Imu::SharedPtr msg) {this->ImuCallback(msg);}
    );
    this->joy_subscriber = ros2_node->create_subscription<sensor_msgs::msg::Joy>(
        TOPIC_JOY, rclcpp::SystemDefaultsQoS(),
        [this] (const sensor_msgs::msg::Joy::SharedPtr msg) {this->JoyCallback(msg);}
    );

    // cmd_vel subscriber
    this->cmd_vel_subscriber = ros2_node->create_subscription<geometry_msgs::msg::Twist>(
        TOPIC_CMD_VEL, rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );

    // height_scan subscriber
    this->height_scan_subscriber = ros2_node->create_subscription<std_msgs::msg::Float32MultiArray>(
        TOPIC_HEIGHT_SCAN, rclcpp::SystemDefaultsQoS(),
        [this] (const std_msgs::msg::Float32MultiArray::SharedPtr msg) {this->HeightScanCallback(msg);}
    );

    // Latched current-FSM-state publisher (transient_local + keep_last(1) so
    // late-joining subscribers immediately receive the most recent state).
    {
        const auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
        this->fsm_state_publisher = ros2_node->create_publisher<robot_msgs::msg::FsmState>(
            "/fsm_state", latched_qos);
    }

    // FSM state-change service.
    this->fsm_set_state_service = ros2_node->create_service<robot_msgs::srv::SetFsmState>(
        "/set_fsm_state",
        [this] (const std::shared_ptr<robot_msgs::srv::SetFsmState::Request>  req,
                      std::shared_ptr<robot_msgs::srv::SetFsmState::Response> res) {
            const std::string& name = req->state_name;
            if (!this->fsm.HasState(name)) {
                res->success = false;
                res->message = "Unknown FSM state: '" + name + "'";
                return;
            }
            const auto& cur = this->fsm.current_state_;
            if (cur && cur->GetStateName() == name) {
                res->success = true;
                res->message = "Already in '" + name + "'";
                return;
            }
            if (cur) {
                auto decision = cur->CanTransitionTo(name);
                if (!decision.allowed) {
                    res->success = false;
                    res->message = decision.reason;
                    return;
                }
            }
            this->fsm.RequestStateChange(name);
            res->success = true;
            res->message = "Transition to '" + name + "' queued";
        });

    // Wire FSM transitions to the latched current-state topic. Also publish the
    // initial state once, since the FSM's initial Enter() ran inside CreateFSM
    // before this callback was registered.
    this->fsm.SetTransitionCallback(
        [this] (const std::string& current_state) {
            robot_msgs::msg::FsmState msg;
            msg.stamp = ros2_node->now();
            msg.current_state = current_state;
            this->fsm_state_publisher->publish(msg);
        });
    if (this->fsm.current_state_) {
        robot_msgs::msg::FsmState initial_msg;
        initial_msg.stamp = ros2_node->now();
        initial_msg.current_state = this->fsm.current_state_->GetStateName();
        this->fsm_state_publisher->publish(initial_msg);
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
    this->plot_real_joint_pos.resize(num_of_dofs);
    this->plot_target_joint_pos.resize(num_of_dofs);
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
    this->loop_rl->shutdown();
    this->loop_control->shutdown();
    this->loop_keyboard->shutdown();
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::OnConfigSwitched()
{
    const float rl_period = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    if (this->loop_rl) this->loop_rl->SetPeriod(rl_period);
    // Intentionally not retuning loop_control — see NOTES.md for why dt should
    // be a robot-level property, not a policy-level one.
}

void RL_Real::GetState(RobotState<float> *state)
{
    std::lock_guard<std::mutex> lock(this->state_mutex);

    // IMU: sensor_msgs/Imu orientation is (x,y,z,w); rl_sdk expects [w,x,y,z].
    state->imu.quaternion[0] = this->imu_msg.orientation.w;
    state->imu.quaternion[1] = this->imu_msg.orientation.x;
    state->imu.quaternion[2] = this->imu_msg.orientation.y;
    state->imu.quaternion[3] = this->imu_msg.orientation.z;

    state->imu.gyroscope[0] = this->imu_msg.angular_velocity.x;
    state->imu.gyroscope[1] = this->imu_msg.angular_velocity.y;
    state->imu.gyroscope[2] = this->imu_msg.angular_velocity.z;

    // Joint feedback: bridge publishes a fixed MotorCmd[12] in physical/SDK order
    // (FR/FL/RL/RR × hip/thigh/calf). joint_mapping[i] gives the bridge index for
    // the i-th policy joint, so per-policy reorderings (e.g. dreamwaq's FL/FR/RL/RR)
    // are handled transparently.
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        const int bridge_idx = joint_mapping[i];
        state->motor_state.q[i]       = this->latest_motor_feedback.motors[bridge_idx].pos;
        state->motor_state.dq[i]      = this->latest_motor_feedback.motors[bridge_idx].vel;
        state->motor_state.tau_est[i] = this->latest_motor_feedback.motors[bridge_idx].torque;
    }
}

void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        auto& m = this->motor_setpoints_msg.motors[joint_mapping[i]];
        m.pos    = command->motor_command.q[i];
        m.vel    = command->motor_command.dq[i];
        m.torque = command->motor_command.tau[i];
        m.kp     = command->motor_command.kp[i];
        m.kw     = command->motor_command.kd[i];
        m.mode   = 0;
    }
    this->motor_setpoints_msg.header.stamp = ros2_node->now();
    this->motor_setpoints_publisher->publish(this->motor_setpoints_msg);
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
        // Serialise inference→push against SwitchToConfig/SwitchToBase. Skip
        // this tick if a switch is in progress.
        std::unique_lock<std::mutex> lock(this->output_mutex, std::try_to_lock);
        if (!lock.owns_lock()) return;

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
        this->plot_real_joint_pos[i].push_back(this->robot_state.motor_state.q[i]);
        this->plot_target_joint_pos[i].push_back(this->robot_command.motor_command.q[i]);
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

void RL_Real::MotorFeedbackCallback(
    const quickbot_interface::msg::MotorFeedback::SharedPtr msg
)
{
    std::lock_guard<std::mutex> lock(this->state_mutex);
    this->latest_motor_feedback = *msg;
    this->joint_state_received = true;
}

void RL_Real::ImuCallback(
    const sensor_msgs::msg::Imu::SharedPtr msg
)
{
    std::lock_guard<std::mutex> lock(this->state_mutex);
    this->imu_msg = *msg;
}

void RL_Real::JoyCallback(
    const sensor_msgs::msg::Joy::SharedPtr msg
)
{
    this->joy_msg = *msg;

    // Layout (RC controller):
    // |__ buttons[]: estop=0 (NO TOUCH, handled at lower level),
    //                2-pole nav mode toggle=1 (0=off, 1=on),
    //                right 3-pole fsm_state=2 (0=GetDown, 1=GetUp, 2=Locomotion),
    //                left 3-pole policy selector=3 (0=none, 1=Dreamwaq, 2=DreamwaqSpeedy; consulted only when btn(2)==2),
    //                back momentary btn=4 (no-op for now)
    // |__ axes[]:    Lx=3, Ly=2, Rx=0, Ry=1
    // Bounds-checked accessors so a controller with fewer axes/buttons can't OOB-index.
    auto btn_val = [&](size_t i) { return i < this->joy_msg.buttons.size() ? this->joy_msg.buttons[i] : 0; };
    auto axis    = [&](size_t i) { return i < this->joy_msg.axes.size() ? this->joy_msg.axes[i] : 0.0f; };

    // btn(1) 2-pole nav mode: drive the level-state bool directly so 1→0 disables, 0→1 enables.
    this->control.navigation_mode = (btn_val(1) != 0);

    // btn(2) drives FSM mode; in the Locomotion position, btn(3) selects which policy.
    // Re-issued every tick — safe because SetGamepad is idempotent and CheckChange is level-triggered.
    switch (btn_val(2))
    {
    case 0: this->control.SetGamepad(Input::Gamepad::B); break; // → GetDown
    case 1: this->control.SetGamepad(Input::Gamepad::A); break; // → GetUp
    case 2:
        switch (btn_val(3))
        {
        case 0: this->control.SetGamepad(Input::Gamepad::A); break;            // no-policy fallback → GetUp
        case 1: this->control.SetGamepad(Input::Gamepad::RB_DPadUp); break;    // → Dreamwaq
        case 2: this->control.SetGamepad(Input::Gamepad::RB_DPadRight); break; // → DreamwaqSpeedy
        }
        break;
    }

    auto joystick_scale = this->params.Get<std::vector<float>>("joystick_scale", {1.0f, 1.0f, 1.0f, 1.0f});
    this->control.x = axis(2) * joystick_scale[1];   // LY
    this->control.y = axis(3) * joystick_scale[0];   // LX
    this->control.yaw = axis(0) * joystick_scale[2]; // RX
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
    // Resize to the incoming array so a config switch can't leave the buffer
    // wrong or overrun the original 187 slots.
    if (this->height_scan_obs.size() != msg->data.size())
    {
        this->height_scan_obs.assign(msg->data.size(), 0.0f);
    }
    std::copy(msg->data.begin(), msg->data.end(), this->height_scan_obs.begin());
    this->last_height_scan_time = ros2_node->now().seconds();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    std::signal(SIGINT, SigintHandler);

    auto rl_sar = std::make_shared<RL_Real>(argc, argv);
    while (rclcpp::ok() && !g_sigint_requested.load())
    {
        rclcpp::spin_some(rl_sar->ros2_node);
    }

    rl_sar.reset();

    rclcpp::shutdown();
    return 0;
}
