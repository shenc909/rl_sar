/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FSM_LOCOMOTION_COMMON_HPP
#define FSM_LOCOMOTION_COMMON_HPP

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
        }
    }

    void Run() override
    {
        if (!rl.rl_init_done) rl.rl_init_done = true;
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO
                  << "RL Controller [" << rl.config_name << "] x:" << rl.control.x
                  << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::endl;
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
};

#endif // FSM_LOCOMOTION_COMMON_HPP
