/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sdk.hpp"

void RL::StateController(const RobotState<float>* state, RobotCommand<float>* command)
{
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;
            rl_fsm_state->fsm_command = command;
        }
    };
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    fsm.Run();

    this->motiontime++;

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    }
}

std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;

    for (const std::string &observation : this->params.Get<std::vector<std::string>>("observations"))
    {
        // ============= Base Observations =============
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.Get<float>("lin_vel_scale"));
        }
        else if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo, mujoco and real robot, the coordinate system for angular velocity is in the body coordinate system.
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->params.Get<float>("ang_vel_scale"));
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.Get<float>("ang_vel_scale"));
            }
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        else if (observation == "commands")
        {
            // Deadband / rate limit / tracking map / range clamp / lateral-accel cap; each stage is a no-op
            // unless the config sets it. Advances the rate limiter by one policy step.
            const float policy_dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
            std::vector<float> commands = this->command_shaper.Step(this->obs.commands, policy_dt);
            obs_list.push_back(commands * this->params.Get<std::vector<float>>("commands_scale"));
        }
        else if (observation == "height_cmd")
        {
            obs_list.push_back(this->obs.height_cmd * this->params.Get<float>("height_cmd_scale", 1.0f));
        }
        else if (observation == "dof_pos")
        {
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * this->params.Get<float>("dof_pos_scale"));
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.Get<float>("dof_vel_scale"));
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
        else if (observation == "height_scan")
        {
            obs_list.push_back(this->obs.height_scan);
        }
        // ============= Other Observations =============
        else if (observation == "whole_body_tracking/motion_command")
        {
            std::vector<float> motion_cmd;
            if (this->motion_loader)
            {
                auto joint_pos_sdk = this->motion_loader->GetJointPos();
                auto joint_vel_sdk = this->motion_loader->GetJointVel();
                auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
                std::vector<float> joint_pos_training(joint_mapping.size());
                std::vector<float> joint_vel_training(joint_mapping.size());
                for (size_t i = 0; i < joint_mapping.size(); ++i)
                {
                    joint_pos_training[i] = joint_pos_sdk[joint_mapping[i]];
                    joint_vel_training[i] = joint_vel_sdk[joint_mapping[i]];
                }
                motion_cmd.insert(motion_cmd.end(), joint_pos_training.begin(), joint_pos_training.end());
                motion_cmd.insert(motion_cmd.end(), joint_vel_training.begin(), joint_vel_training.end());
            }
            else
            {
                motion_cmd.resize(this->params.Get<int>("num_of_dofs") * 2, 0.0f);
            }
            obs_list.push_back(motion_cmd);
        }
        else if (observation == "whole_body_tracking/motion_anchor_ori_b")
        {
            std::vector<float> anchor_ori(6, 0.0f);
            if (this->motion_loader)
            {
                auto waist_sdk_indices = this->params.Get<std::vector<int>>("waist_joint_indices");
                std::vector<float> waist_angles = {
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[0])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[1])],
                    this->obs.dof_pos[InverseJointMapping(waist_sdk_indices[2])]
                };
                std::vector<float> robot_torso_quat_w = MotionLoader::ComputeTorsoQuat(this->obs.base_quat, waist_angles);
                std::vector<float> ref_torso_quat_w = this->motion_loader->GetAnchorQuat();
                std::vector<float> init_quat = this->motion_loader->GetInitQuat();
                std::vector<float> motion_anchor_quat_w = QuaternionMultiply(init_quat, ref_torso_quat_w);
                std::vector<float> robot_quat_inv = QuaternionConjugate(robot_torso_quat_w);
                std::vector<float> relative_quat = QuaternionMultiply(robot_quat_inv, motion_anchor_quat_w);
                std::vector<float> rot_matrix = QuaternionToRotationMatrix(relative_quat);
                anchor_ori = MatrixFirstTwoColumns(rot_matrix);
            }
            obs_list.push_back(anchor_ori);
        }
        else if (observation == "RoboMimic_Deploy/phase")
        {
            float motion_time = this->episode_length_buf * this->params.Get<float>("dt") * this->params.Get<int>("decimation");
            float count = motion_time;
            float phase = count / this->motion_length;
            std::vector<float> phase_vec = {phase};
            obs_list.push_back(phase_vec);
        }
    }

    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    std::vector<float> clamped_obs = clamp(obs, -this->params.Get<float>("clip_obs"), this->params.Get<float>("clip_obs"));
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.height_cmd = {this->params.Get<float>("height_cmd_default", 0.0f)};
    this->obs.base_quat = {0.0f, 0.0f, 0.0f, 1.0f};
    this->obs.dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.actions.clear();
    this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->obs.height_scan = this->MakeHeightScanFill();
    this->ConfigureCommandShaper();
    this->ComputeObservation();
}

void RL::ConfigureCommandShaper()
{
    const size_t num_axes = this->obs.commands.size();
    CommandShaper::Config config;
    std::vector<std::string> problems;
    auto per_axis = [&](const std::string& key) {
        auto v = this->params.Get<std::vector<float>>(key);
        if (!v.empty() && v.size() != num_axes)
        {
            problems.push_back(key + " needs " + std::to_string(num_axes) + " values");
            v.clear();
        }
        return v;
    };

    config.deadband = per_axis("command_deadband");
    config.max_accel = per_axis("command_max_accel");
    config.max_decel = per_axis("command_max_decel");
    if (!config.max_decel.empty() && config.max_accel.empty())
    {
        problems.push_back("command_max_decel has no effect without command_max_accel");
    }

    // One "command_tracking_map_<axis>" list of [desired, commanded] points per axis.
    const std::vector<std::string> axis_names = {"vx", "vy", "wz"};
    std::vector<CommandShaper::Curve> tracking_map(num_axes);
    bool any_map = false;
    for (size_t i = 0; i < num_axes && i < axis_names.size(); ++i)
    {
        const std::string key = "command_tracking_map_" + axis_names[i];
        const auto points = this->params.Get<std::vector<std::vector<float>>>(key);
        if (points.empty()) continue;
        tracking_map[i] = CommandShaper::MakeCurve(points);
        if (tracking_map[i].empty())
        {
            problems.push_back(key + " must be [desired, commanded] pairs with distinct desired values");
            continue;
        }
        any_map = true;
    }
    if (any_map) config.tracking_map = tracking_map;

    config.range_lower = per_axis("command_range_lower");
    config.range_upper = per_axis("command_range_upper");
    if (config.range_lower.empty() != config.range_upper.empty())
    {
        problems.push_back("command_range_lower and command_range_upper must be set together");
        config.range_lower.clear();
        config.range_upper.clear();
    }
    for (size_t i = 0; i < config.range_lower.size(); ++i)
    {
        if (config.range_lower[i] > config.range_upper[i])
        {
            problems.push_back("command_range_lower exceeds command_range_upper on axis " + std::to_string(i));
            config.range_lower.clear();
            config.range_upper.clear();
            break;
        }
    }

    config.max_lateral_accel = this->params.Get<float>("max_lateral_accel", 0.0f);

    for (const auto& problem : problems)
    {
        std::cout << LOGGER::WARNING << "Command shaper: " << problem << "; stage disabled." << std::endl;
    }
    this->command_shaper.Configure(config, num_axes);
}

std::vector<float> RL::MakeHeightScanFill() const
{
    const int num_points = this->params.Get<int>("num_height_scan_points");
    const auto fill = this->params.Get<std::vector<float>>("height_scan_fill");
    if (fill.empty())
    {
        // No per-channel fill configured: keep the legacy uniform placeholder.
        return std::vector<float>(num_points, 0.31f);
    }
    // Split num_points into "height_scan_num_channels" equal contiguous (channel-major) segments,
    // filling segment i with fill[i]. Falls back to fill.size() channels if the count is unset.
    const int num_channels = this->params.Get<int>("height_scan_num_channels", static_cast<int>(fill.size()));
    if (num_channels <= 0 || static_cast<int>(fill.size()) != num_channels || num_points % num_channels != 0)
    {
        std::cout << LOGGER::WARNING << "height_scan_fill (" << fill.size() << " values) / height_scan_num_channels ("
                  << num_channels << ") inconsistent with num_height_scan_points (" << num_points
                  << "); falling back to uniform 0.31 fill." << std::endl;
        return std::vector<float>(num_points, 0.31f);
    }
    const int per_channel = num_points / num_channels;
    std::vector<float> height_scan(num_points);
    for (int c = 0; c < num_channels; ++c)
    {
        std::fill(height_scan.begin() + c * per_channel, height_scan.begin() + (c + 1) * per_channel, fill[c]);
    }
    return height_scan;
}

void RL::InitOutputs()
{
    int num_of_dofs = this->params.Get<int>("num_of_dofs");
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);
    this->output_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
}

void RL::InitControl()
{
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
    // Called from InitRL on every policy entry, so each (re)entry starts at the nominal height.
    this->control.height = this->params.Get<float>("height_cmd_default", 0.0f);
}

void RL::UpdateHeightCommand(float stick, float dt)
{
    if (!this->params.Has("height_cmd_range")) return;
    const float deadzone = std::clamp(this->params.Get<float>("height_cmd_deadzone", 0.1f), 0.0f, 0.99f);
    float s = std::clamp(stick, -1.0f, 1.0f);
    if (std::fabs(s) <= deadzone) return;
    // Rescale past the deadzone so the rate ramps continuously from 0 instead of jumping.
    s = (s - std::copysign(deadzone, s)) / (1.0f - deadzone);
    this->SetHeightCommand(this->control.height + s * this->params.Get<float>("height_cmd_rate", 0.2f) * dt);
}

void RL::SetHeightCommand(float value)
{
    const auto range = this->params.Get<std::vector<float>>("height_cmd_range");
    if (range.size() != 2) return;
    this->control.height = std::clamp(value, range[0], range[1]);
}

void RL::InitJointNum(size_t num_joints)
{
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);
    this->robot_command.motor_command.resize(num_joints);
}

void RL::InitRL(std::string robot_config_path)
{
    std::lock_guard<std::mutex> lock(this->model_mutex);

    this->ReadYaml(robot_config_path, "config.yaml");
    this->AutoGenerateJointMapping();

    // init joint num first
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));

    // init rl
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();

    // init obs history
    const auto& observations_history = this->params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
    if (!observations_history.empty())
    {
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->params.Get<std::string>("observations_history_priority"));
    }

    // init model
    std::string model_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("model_name");
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }
}

void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    {
        pos_actions_scaled[i] = 0.0f;
        vel_actions_scaled[i] = actions_scaled[i];
    }
    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    output_dof_vel = vel_actions_scaled;
    output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (all_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) - this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
    output_dof_tau = clamp(output_dof_tau, -this->params.Get<std::vector<float>>("torque_limits"), this->params.Get<std::vector<float>>("torque_limits"));
}

int RL::InverseJointMapping(int idx) const
{
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}

// Take values authored against `old_jm` and rewrite them in-place under `new_jm`,
// preserving the physical/SDK-indexed values across the swap.
static void RemapJointVector(std::vector<float>& v,
                             const std::vector<int>& old_jm,
                             const std::vector<int>& new_jm)
{
    const int n = static_cast<int>(old_jm.size());
    if (static_cast<int>(v.size()) != n) return;
    std::vector<float> phys(n, 0.0f);
    for (int i = 0; i < n; ++i) phys[old_jm[i]] = v[i];
    for (int i = 0; i < n; ++i) v[i] = phys[new_jm[i]];
}

void RL::SwitchToConfig(const std::string& robot_config_path)
{
    // Hold output_mutex for the *entire* switch so RunModel either:
    //   (a) hadn't started its inference cycle yet — its try-to-lock will fail
    //       while we're swapping params; the cycle is skipped; or
    //   (b) was already mid-cycle when we tried to switch — its lock prevents
    //       us from advancing until it finishes its push. The push lands in
    //       the queue under OLD params/order, but our drain below clears it
    //       before any post-switch RunModel sees the queue.
    std::lock_guard<std::mutex> swap_lock(this->output_mutex);

    auto old_jm = this->params.Get<std::vector<int>>("joint_mapping");

    // Reset the params node and re-base. Without this, keys present in the
    // previous policy's config but absent from the new one leak through
    // (e.g. dreamwaq's `explicit_joint_names` survives into robot_lab and
    // makes AutoGenerateJointMapping override robot_lab's [0..11] joint_mapping
    // with dreamwaq's permutation, scattering joint commands).
    this->params.config_node = YAML::Node();
    this->ReadYaml(this->robot_name, "base.yaml");

    this->InitRL(robot_config_path);
    auto new_jm = this->params.Get<std::vector<int>>("joint_mapping");

    // Drain output queues — any items present here are stale: either pushed by
    // the previous policy in OLD training order, or pushed by an in-flight
    // cycle that finished before we acquired output_mutex.
    {
        std::vector<float> tmp;
        while (this->output_dof_pos_queue.try_pop(tmp)) {}
        while (this->output_dof_vel_queue.try_pop(tmp)) {}
        while (this->output_dof_tau_queue.try_pop(tmp)) {}
    }

    if (old_jm != new_jm)
    {
        auto& mc = this->robot_command.motor_command;
        auto& ms = this->robot_state.motor_state;
        RemapJointVector(mc.q,   old_jm, new_jm);
        RemapJointVector(mc.dq,  old_jm, new_jm);
        RemapJointVector(mc.kp,  old_jm, new_jm);
        RemapJointVector(mc.kd,  old_jm, new_jm);
        RemapJointVector(mc.tau, old_jm, new_jm);
        RemapJointVector(ms.q,   old_jm, new_jm);
        RemapJointVector(ms.dq,  old_jm, new_jm);
        RemapJointVector(ms.tau_est, old_jm, new_jm);
    }
    this->now_state = this->robot_state;
    this->OnConfigSwitched();
}

void RL::SwitchToBase()
{
    // Same rationale as SwitchToConfig: hold output_mutex through the param
    // swap + joint-mapping remap so a concurrent RunModel cycle can't read
    // half-remapped state or push outputs computed against mismatched params.
    std::lock_guard<std::mutex> swap_lock(this->output_mutex);

    auto old_jm = this->params.Get<std::vector<int>>("joint_mapping");
    // Reset params so policy-specific keys from the previously-active config
    // (e.g. explicit_joint_names) don't survive into intermediate states like
    // GetUp / GetDown.
    this->params.config_node = YAML::Node();
    this->ReadYaml(this->robot_name, "base.yaml");
    auto new_jm = this->params.Get<std::vector<int>>("joint_mapping");

    // Drain queues for the same reason as SwitchToConfig.
    {
        std::vector<float> tmp;
        while (this->output_dof_pos_queue.try_pop(tmp)) {}
        while (this->output_dof_vel_queue.try_pop(tmp)) {}
        while (this->output_dof_tau_queue.try_pop(tmp)) {}
    }

    if (old_jm != new_jm)
    {
        auto& mc = this->robot_command.motor_command;
        auto& ms = this->robot_state.motor_state;
        RemapJointVector(mc.q,   old_jm, new_jm);
        RemapJointVector(mc.dq,  old_jm, new_jm);
        RemapJointVector(mc.kp,  old_jm, new_jm);
        RemapJointVector(mc.kd,  old_jm, new_jm);
        RemapJointVector(mc.tau, old_jm, new_jm);
        RemapJointVector(ms.q,   old_jm, new_jm);
        RemapJointVector(ms.dq,  old_jm, new_jm);
        RemapJointVector(ms.tau_est, old_jm, new_jm);
    }
    this->now_state = this->robot_state;
}

void RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[i];
        float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[index];
            float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // Use QuaternionToEuler from vector_math.hpp
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

static int kbhit()
{
    static bool initialized = false;
    static termios original_term;

    // Initialize terminal to non-canonical mode on first call
    if (!initialized)
    {
        tcgetattr(STDIN_FILENO, &original_term);

        termios new_term = original_term;
        new_term.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        new_term.c_cc[VMIN] = 0;   // Non-blocking read
        new_term.c_cc[VTIME] = 0;  // No timeout

        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        // Register cleanup function to restore terminal on exit
        static bool cleanup_registered = false;
        if (!cleanup_registered)
        {
            std::atexit([]() {
                tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
            });
            cleanup_registered = true;
        }

        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(STDIN_FILENO, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}

void RL::KeyboardInterface()
{
    int c = kbhit();
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(STDIN_FILENO, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
    }
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

namespace
{
int64_t SteadyNowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Resolves SDL GameController buttons to one gamepad command with the same precedence as
// rl_sim's /joy callback: the last match wins, so combos beat singles and LB+RB beats all.
Input::Gamepad ResolveSdlGamepad(const std::vector<int> &buttons)
{
    using G = Input::Gamepad;
    enum : size_t { A = 0, B = 1, X = 2, Y = 3, LS = 7, RS = 8, LB = 9, RB = 10, UP = 11, DOWN = 12, LEFT = 13, RIGHT = 14 };
    auto btn = [&](size_t i) { return i < buttons.size() && buttons[i] != 0; };
    static const std::vector<std::pair<size_t, G>> singles = {
        {A, G::A}, {B, G::B}, {X, G::X}, {Y, G::Y}, {LB, G::LB}, {RB, G::RB}, {LS, G::LStick}, {RS, G::RStick},
        {UP, G::DPadUp}, {DOWN, G::DPadDown}, {LEFT, G::DPadLeft}, {RIGHT, G::DPadRight}};
    static const std::vector<std::pair<size_t, G>> lb_combos = {
        {A, G::LB_A}, {B, G::LB_B}, {X, G::LB_X}, {Y, G::LB_Y}, {LS, G::LB_LStick}, {RS, G::LB_RStick},
        {UP, G::LB_DPadUp}, {DOWN, G::LB_DPadDown}, {LEFT, G::LB_DPadLeft}, {RIGHT, G::LB_DPadRight}};
    static const std::vector<std::pair<size_t, G>> rb_combos = {
        {A, G::RB_A}, {B, G::RB_B}, {X, G::RB_X}, {Y, G::RB_Y}, {LS, G::RB_LStick}, {RS, G::RB_RStick},
        {UP, G::RB_DPadUp}, {DOWN, G::RB_DPadDown}, {LEFT, G::RB_DPadLeft}, {RIGHT, G::RB_DPadRight}};

    G result = G::None;
    for (const auto &entry : singles) if (btn(entry.first)) result = entry.second;
    if (btn(LB)) for (const auto &entry : lb_combos) if (btn(entry.first)) result = entry.second;
    if (btn(RB)) for (const auto &entry : rb_combos) if (btn(entry.first)) result = entry.second;
    if (btn(LB) && btn(RB)) result = G::LB_RB;
    return result;
}
}  // namespace

void RL::AppJoyCallback(const std::vector<float> &axes, const std::vector<int> &buttons)
{
    auto axis = [&](size_t i) { return i < axes.size() ? axes[i] : 0.0f; };

    if (!this->app_joy_active.load())
    {
        // Fresh start: a button already held when the app (re)appears counts as a new press.
        this->app_joy_last_combo = Input::Gamepad::None;
        std::cout << LOGGER::INFO << "[AppJoy] /joy_app active, app has the sticks" << std::endl;
    }
    const Input::Gamepad combo = ResolveSdlGamepad(buttons);
    if (combo != this->app_joy_last_combo && combo != Input::Gamepad::None)
    {
        this->app_joy_pending_combo.store(static_cast<int>(combo));
    }
    this->app_joy_last_combo = combo;

    std::vector<float> scale;
    {
        std::lock_guard<std::mutex> lock(this->app_joy_params_mutex);
        scale = this->app_joy_scale;
    }
    this->control.x = axis(1) * scale[1];    // LY
    this->control.y = axis(0) * scale[0];    // LX
    this->control.yaw = axis(2) * scale[2];  // RX
    this->app_joy_height_axis.store(axis(3)); // RY: integrated by UpdateHeightCommand
    this->app_joy_last_ns.store(SteadyNowNs());
    this->app_joy_active.store(true);
}

void RL::AppJoyTick()
{
    const auto pending = static_cast<Input::Gamepad>(
        this->app_joy_pending_combo.exchange(static_cast<int>(Input::Gamepad::None)));
    if (pending != Input::Gamepad::None)
    {
        this->control.SetGamepad(pending);
        std::cout << LOGGER::INFO << "[AppJoy] " << Input::GamepadName(pending) << std::endl;
    }

    if (!this->app_joy_active.load()) return;
    float timeout;
    {
        std::lock_guard<std::mutex> lock(this->app_joy_params_mutex);
        timeout = this->app_joy_timeout;
    }
    if (SteadyNowNs() - this->app_joy_last_ns.load() > static_cast<int64_t>(timeout * 1e9))
    {
        this->app_joy_active.store(false);
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->app_joy_height_axis.store(0.0f);
        std::cout << LOGGER::WARNING << "[AppJoy] No /joy_app for " << timeout << " s, sticks zeroed" << std::endl;
    }
}

void RL::JointStateTick()
{
    // physical_joint_names is fixed after ReadYaml; joint_mapping can change with a policy switch, which
    // happens on this (the control) thread, so reading params here is safe.
    if (this->physical_joint_names.empty()) return;
    const auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    const size_t n = this->physical_joint_names.size();
    std::vector<double> position(n, 0.0), velocity(n, 0.0);
    // motor_state is in policy order: entry i is physical joint joint_mapping[i].
    for (size_t i = 0; i < joint_mapping.size() && i < this->robot_state.motor_state.q.size(); ++i)
    {
        const int physical = joint_mapping[i];
        if (physical < 0 || static_cast<size_t>(physical) >= n) continue;
        position[physical] = this->robot_state.motor_state.q[i];
        velocity[physical] = this->robot_state.motor_state.dq[i];
    }
    std::lock_guard<std::mutex> lock(this->joint_state_mutex);
    if (this->joint_state_names.size() != n) this->joint_state_names = this->physical_joint_names;
    this->joint_state_position.swap(position);
    this->joint_state_velocity.swap(velocity);
}

void RL::RequestNavigationMode(bool enabled)
{
    this->navigation_mode_request.store(enabled ? 1 : 0);
}

void RL::NavigationModeTick()
{
    const int request = this->navigation_mode_request.exchange(-1);
    if (request >= 0 && this->control.navigation_mode != (request == 1))
    {
        this->control.navigation_mode = (request == 1);
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
    }
    // Last tick's X / N toggle shows up here too, one tick (a few ms) late.
    this->navigation_mode_snapshot.store(this->control.navigation_mode);
}

bool RL::GetJointStateSnapshot(std::vector<std::string> &names, std::vector<double> &position, std::vector<double> &velocity)
{
    std::lock_guard<std::mutex> lock(this->joint_state_mutex);
    if (this->joint_state_names.empty()) return false;
    names = this->joint_state_names;
    position = this->joint_state_position;
    velocity = this->joint_state_velocity;
    return true;
}

void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }

    if (file_name == "base.yaml")
    {
        this->physical_joint_names = this->params.Get<std::vector<std::string>>("joint_names");
    }

    {
        auto scale = this->params.Get<std::vector<float>>("joystick_scale", {1.0f, 1.0f, 1.0f, 1.0f});
        scale.resize(4, 1.0f);
        std::lock_guard<std::mutex> lock(this->app_joy_params_mutex);
        this->app_joy_scale = scale;
        this->app_joy_timeout = this->params.Get<float>("app_joy_timeout", 0.5f);
    }
}

void RL::AutoGenerateJointMapping()
{
    if (this->physical_joint_names.empty()) return;

    auto cfg_names = this->params.Get<std::vector<std::string>>("explicit_joint_names");
    if (cfg_names.empty()) return;  // config did not define an explicit policy ordering

    int num_of_dofs = this->params.Get<int>("num_of_dofs");
    if (static_cast<int>(cfg_names.size()) != num_of_dofs)
    {
        throw std::runtime_error(
            "[joint_mapping] config.yaml explicit_joint_names size (" + std::to_string(cfg_names.size()) +
            ") does not match num_of_dofs (" + std::to_string(num_of_dofs) + ")");
    }

    std::vector<int> mapping;
    mapping.reserve(cfg_names.size());
    for (const auto& name : cfg_names)
    {
        auto it = std::find(this->physical_joint_names.begin(), this->physical_joint_names.end(), name);
        if (it == this->physical_joint_names.end())
        {
            throw std::runtime_error(
                "[joint_mapping] joint '" + name + "' from config.yaml explicit_joint_names is not present in base.yaml's joint_names");
        }
        mapping.push_back(static_cast<int>(std::distance(this->physical_joint_names.begin(), it)));
    }

    YAML::Node node;
    for (int v : mapping) node.push_back(v);
    this->params.config_node["joint_mapping"] = node;

    std::cout << LOGGER::INFO << "[joint_mapping] auto-derived from config.yaml explicit_joint_names: [";
    for (size_t i = 0; i < mapping.size(); ++i)
    {
        std::cout << mapping[i] << (i + 1 == mapping.size() ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

void RL::CSVInit(std::string robot_path)
{
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_cal_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_est_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_target_" << i << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_vel_" << i << ","; }

    file << std::endl;

    file.close();
}

void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}

bool RLFSMState::Interpolate(
    float& percent,
    const std::vector<float>& start_pos,
    const std::vector<float>& target_pos,
    float duration_seconds,
    const std::string& description,
    bool use_fixed_gains)
{
    if (percent >= 1.0f)
    {
        return false;
    }

    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / rl.params.Get<float>("dt"))));
    float step = 1.0f / required_frames;

    percent += step;
    percent = std::min(percent, 1.0f);

    auto kp = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kp") : rl.params.Get<std::vector<float>>("rl_kp");
    auto kd = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kd") : rl.params.Get<std::vector<float>>("rl_kd");

    for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
    {
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;
    }

    if (!description.empty())
    {
        LOGGER::PrintProgress(percent, description);
    }

    if (percent >= 1.0f)
    {
        return false;
    }

    return true;
}

void RLFSMState::RLControl()
{
    std::vector<float> _output_dof_pos, _output_dof_vel;
    if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            if (!_output_dof_pos.empty())
            {
                fsm_command->motor_command.q[i] = _output_dof_pos[i];
            }
            if (!_output_dof_vel.empty())
            {
                fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            }
            fsm_command->motor_command.kp[i] = rl.params.Get<std::vector<float>>("rl_kp")[i];
            fsm_command->motor_command.kd[i] = rl.params.Get<std::vector<float>>("rl_kd")[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}
