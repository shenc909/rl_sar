/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FSM_LOCOMOTION_COMMON_HPP
#define FSM_LOCOMOTION_COMMON_HPP

#include <algorithm>
#include <cmath>

#include "fsm.hpp"
#include "rl_sdk.hpp"

// Header-only base class for any "subclass = one locomotion policy" pattern.
// Concrete subclasses (declared in per-robot fsm_<robot>.hpp files) supply:
//   - their FSM state name
//   - the policy folder name under policy/<robot>/
//   - optional Enter() override (e.g. a custom default-pose interpolation
//     before the policy switch)
//   - optional CheckChange() override that lists this state's legal
//     inter-policy transitions and falls through to
//     RLFSMStateRLLocomotion::CheckChange() for the universal exits.
//   - optional CanTransitionTo() override widening the legal targets seen by
//     external (non-input-driven) requesters such as the ROS service.
class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL* rl, std::string state_name, std::string config_name)
        : RLFSMState(*rl, std::move(state_name)),
          config_name_(std::move(config_name)) {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        rl.config_name = config_name_;
        pre_running_percent_ = 0.0f;
        try
        {
            rl.SwitchToConfig(rl.robot_name + "/" + rl.config_name);
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
            return;
        }
        transition_duration_ = rl.params.Get<float>("transition_duration", 2.0f);
    }

    void Run() override
    {
        if (!rl.rl_init_done) rl.rl_init_done = true;
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO
                  << "RL Controller [" << rl.config_name << "] x:" << rl.control.x
                  << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::endl;

        // Pre-RL takeover (port of robot-ros2's pattern): the percent only
        // advances when a fresh policy output is consumed, q is blended from
        // init_dof_pos (config-defined) toward the live policy output, and
        // kp/kd snap to rl_kp/rl_kd as soon as the queue feeds us. While the
        // queue is empty we leave fsm_command alone so the held GetUp pose
        // continues to be commanded.
        std::vector<float> _output_dof_pos, _output_dof_vel;
        if (!rl.output_dof_pos_queue.try_pop(_output_dof_pos) ||
            !rl.output_dof_vel_queue.try_pop(_output_dof_vel))
        {
            return;
        }

        const auto rl_kp = rl.params.Get<std::vector<float>>("rl_kp");
        const auto rl_kd = rl.params.Get<std::vector<float>>("rl_kd");
        const int n = rl.params.Get<int>("num_of_dofs");

        if (pre_running_percent_ < 1.0f)
        {
            // Ramp pace: complete in transition_duration seconds at the
            // inference period (dt * decimation), since each successful pop
            // corresponds to one inference cycle.
            const float rl_period = rl.params.Get<float>("dt") * rl.params.Get<int>("decimation");
            const int n_steps = std::max(1, static_cast<int>(std::ceil(transition_duration_ / rl_period)));
            pre_running_percent_ = std::min(pre_running_percent_ + 1.0f / n_steps, 1.0f);

            const auto init_dof_pos = rl.params.Get<std::vector<float>>("init_dof_pos",
                rl.params.Get<std::vector<float>>("default_dof_pos"));
            const float a = pre_running_percent_;
            for (int i = 0; i < n; ++i)
            {
                if (!_output_dof_pos.empty())
                    fsm_command->motor_command.q[i] = (1 - a) * init_dof_pos[i] + a * _output_dof_pos[i];
                fsm_command->motor_command.dq[i]  = 0;
                fsm_command->motor_command.kp[i]  = rl_kp[i];
                fsm_command->motor_command.kd[i]  = rl_kd[i];
                fsm_command->motor_command.tau[i] = 0;
            }
        }
        else
        {
            for (int i = 0; i < n; ++i)
            {
                if (!_output_dof_pos.empty())
                    fsm_command->motor_command.q[i] = _output_dof_pos[i];
                if (!_output_dof_vel.empty())
                    fsm_command->motor_command.dq[i] = _output_dof_vel[i];
                fsm_command->motor_command.kp[i]  = rl_kp[i];
                fsm_command->motor_command.kd[i]  = rl_kd[i];
                fsm_command->motor_command.tau[i] = 0;
            }
        }
    }

    void Exit() override
    {
        rl.SwitchToBase();
        rl.rl_init_done = false;
    }

    // Default: only handle the universal exits. Subclasses extend by checking
    // their inter-policy transitions first, then
    // `return RLFSMStateRLLocomotion::CheckChange();` as the fallthrough.
    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
            return "RLFSMStatePassive";
        if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            return "RLFSMStateGetDown";
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
            return "RLFSMStateGetUp";
        return state_name_;
    }

    // Default legality for any locomotion state: universal exits and self-stay.
    // Subclasses widen by listing their inter-policy alternates.
    ChangeDecision CanTransitionTo(const std::string& target) const override
    {
        if (target == "RLFSMStatePassive" ||
            target == "RLFSMStateGetUp" ||
            target == "RLFSMStateGetDown" ||
            target == GetStateName())
        {
            return {true, ""};
        }
        return {false, "Transition from '" + GetStateName() + "' to '" + target +
                       "' is not in this state's legal targets."};
    }

protected:
    std::string config_name_;   // policy folder under policy/<robot_name>/

private:
    float pre_running_percent_ = 0.0f;
    float transition_duration_ = 2.0f;
};

#endif // FSM_LOCOMOTION_COMMON_HPP
