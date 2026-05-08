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
        // Snapshot the held command (already in the new policy's joint order
        // after SwitchToConfig's RemapJointVector). Run() ramps from these
        // GetUp-era gains/pose to the policy's rl_kp/rl_kd and live output
        // across `transition_duration` seconds. Without this, a single tick
        // jumps kp 80→25 and q by ~0.2 rad on rear thighs, which the real
        // robot expresses as a visible joint snap. Sim hides it because
        // gazebo_ros_control's joint PID and softer dynamics absorb the step.
        const auto& mc = rl.robot_command.motor_command;
        held_q_  = mc.q;
        held_kp_ = mc.kp;
        held_kd_ = mc.kd;
        percent_transition_ = 0.0f;
        transition_duration_ = rl.params.Get<float>("transition_duration", 0.5f);
    }

    void Run() override
    {
        if (!rl.rl_init_done) rl.rl_init_done = true;
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO
                  << "RL Controller [" << rl.config_name << "] x:" << rl.control.x
                  << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::endl;

        if (percent_transition_ < 1.0f)
        {
            BlendToPolicy();
            return;
        }
        RLControl();
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
    // Blend held GetUp gains/pose toward the RL policy's gains/output across
    // `transition_duration_` seconds. Pops policy outputs as soon as they
    // become available so RunModel's queue doesn't pile up; falls back to the
    // held pose for the first few ticks before inference completes.
    void BlendToPolicy()
    {
        std::vector<float> _output_dof_pos, _output_dof_vel;
        const bool have_output = rl.output_dof_pos_queue.try_pop(_output_dof_pos)
                              && rl.output_dof_vel_queue.try_pop(_output_dof_vel);

        const float dt = rl.params.Get<float>("dt");
        const int required_frames = std::max(1,
            static_cast<int>(std::ceil(transition_duration_ / dt)));
        percent_transition_ = std::min(percent_transition_ + 1.0f / required_frames, 1.0f);

        const auto rl_kp = rl.params.Get<std::vector<float>>("rl_kp");
        const auto rl_kd = rl.params.Get<std::vector<float>>("rl_kd");
        const int n = rl.params.Get<int>("num_of_dofs");
        const float a = percent_transition_;

        for (int i = 0; i < n; ++i)
        {
            const float q_target  = have_output ? _output_dof_pos[i] : held_q_[i];
            const float dq_target = have_output ? _output_dof_vel[i] : 0.0f;
            fsm_command->motor_command.q[i]   = (1 - a) * held_q_[i]  + a * q_target;
            fsm_command->motor_command.dq[i]  = a * dq_target;
            fsm_command->motor_command.kp[i]  = (1 - a) * held_kp_[i] + a * rl_kp[i];
            fsm_command->motor_command.kd[i]  = (1 - a) * held_kd_[i] + a * rl_kd[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    std::vector<float> held_q_;
    std::vector<float> held_kp_;
    std::vector<float> held_kd_;
    float percent_transition_ = 0.0f;
    float transition_duration_ = 0.5f;
};

#endif // FSM_LOCOMOTION_COMMON_HPP
