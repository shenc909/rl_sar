/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUICKBOT_FSM_HPP
#define QUICKBOT_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"
#include "fsm_locomotion_common.hpp"

namespace quickbot_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    void Run() override
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = 0;
            fsm_command->motor_command.kd[i] = 8;
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }

    ChangeDecision CanTransitionTo(const std::string& target) const override
    {
        if (target == "RLFSMStateGetUp" || target == GetStateName())
            return {true, ""};
        return {false, "Cannot leave Passive directly to '" + target + "'; go via GetUp first."};
    }
};

class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_pre_getup = 0.0f;
    float percent_getup = 0.0f;
    std::vector<float> pre_running_pos = {
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 1.36, -2.65,
        0.00, 0.00, 0.00, 0.00
    };
    bool stand_from_passive = true;

    void Enter() override
    {
        percent_pre_getup = 0.0f;
        percent_getup = 0.0f;
        if (rl.fsm.previous_state_->GetStateName() == "RLFSMStatePassive")
        {
            stand_from_passive = true;
        }
        else
        {
            stand_from_passive = false;
        }
        rl.now_state = *fsm_state;
        if (stand_from_passive)
        {
            rl.start_state = rl.now_state;
        }
    }

    void Run() override
    {
        if(stand_from_passive)
        {

            if (Interpolate(percent_pre_getup, rl.now_state.motor_state.q, pre_running_pos, 1.0f, "Pre Getting up", true)) return;
            if (Interpolate(percent_getup, pre_running_pos, rl.params.Get<std::vector<float>>("default_dof_pos"), 2.0f, "Getting up", true)) return;
        }
        else
        {
            if (Interpolate(percent_getup, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 1.0f, "Getting up", true)) return;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            {
                return "RLFSMStateGetDown";
            }
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            {
                return "RLFSMStateRLLocomotion_Dreamwaq";
            }
            if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
            {
                return "RLFSMStateRLLocomotion_DreamwaqSpeedy";
            }
        }
        return state_name_;
    }

    ChangeDecision CanTransitionTo(const std::string& target) const override
    {
        if (target == "RLFSMStatePassive" || target == GetStateName())
            return {true, ""};

        const bool is_locomotion = target == "RLFSMStateRLLocomotion_Dreamwaq"
                                || target == "RLFSMStateRLLocomotion_DreamwaqSpeedy";
        if (target == "RLFSMStateGetDown" || is_locomotion)
        {
            if (percent_getup < 1.0f)
            {
                return {false, "GetUp not complete (percent_getup=" + std::to_string(percent_getup) +
                               "); transitions to '" + target + "' not yet allowed."};
            }
            return {true, ""};
        }
        return {false, "Transition from 'RLFSMStateGetUp' to '" + target + "' is not in this state's legal targets."};
    }
};

class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;
    }

    void Run() override
    {
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
        {
            return "RLFSMStatePassive";
        }
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            return "RLFSMStateGetUp";
        }
        return state_name_;
    }

    ChangeDecision CanTransitionTo(const std::string& target) const override
    {
        if (target == "RLFSMStatePassive" || target == "RLFSMStateGetUp" || target == GetStateName())
            return {true, ""};
        return {false, "Transition from 'RLFSMStateGetDown' to '" + target + "' is not in this state's legal targets."};
    }
};

class RLFSMStateRLLocomotion_Dreamwaq : public RLFSMStateRLLocomotion
{
public:
    RLFSMStateRLLocomotion_Dreamwaq(RL *rl)
        : RLFSMStateRLLocomotion(rl,
            "RLFSMStateRLLocomotion_Dreamwaq",
            "dreamwaq") {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num2 || rl.control.current_gamepad == Input::Gamepad::RB_DPadRight)
            return "RLFSMStateRLLocomotion_DreamwaqSpeedy";
        return RLFSMStateRLLocomotion::CheckChange();
    }

    ChangeDecision CanTransitionTo(const std::string& target) const override
    {
        if (target == "RLFSMStateRLLocomotion_DreamwaqSpeedy")
        {
            return {true, ""};
        }
        return RLFSMStateRLLocomotion::CanTransitionTo(target);
    }
};

class RLFSMStateRLLocomotion_DreamwaqSpeedy : public RLFSMStateRLLocomotion
{
public:
    RLFSMStateRLLocomotion_DreamwaqSpeedy(RL *rl)
        : RLFSMStateRLLocomotion(rl,
            "RLFSMStateRLLocomotion_DreamwaqSpeedy",
            "dreamwaq_speedy") {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            return "RLFSMStateRLLocomotion_Dreamwaq";
        return RLFSMStateRLLocomotion::CheckChange();
    }

    ChangeDecision CanTransitionTo(const std::string& target) const override
    {
        if (target == "RLFSMStateRLLocomotion_Dreamwaq")
        {
            return {true, ""};
        }
        return RLFSMStateRLLocomotion::CanTransitionTo(target);
    }
};

} // namespace quickbot_fsm

class QuickbotFSMFactory : public FSMFactory
{
public:
    QuickbotFSMFactory(const std::string& initial) : initial_state_(initial) {}
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        RL *rl = static_cast<RL *>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<quickbot_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<quickbot_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<quickbot_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRLLocomotion_Dreamwaq")
            return std::make_shared<quickbot_fsm::RLFSMStateRLLocomotion_Dreamwaq>(rl);
        else if (state_name == "RLFSMStateRLLocomotion_DreamwaqSpeedy")
            return std::make_shared<quickbot_fsm::RLFSMStateRLLocomotion_DreamwaqSpeedy>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "quickbot"; }
    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLLocomotion_Dreamwaq",
            "RLFSMStateRLLocomotion_DreamwaqSpeedy"
        };
    }
    std::string GetInitialState() const override { return initial_state_; }
private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(QuickbotFSMFactory, "RLFSMStatePassive")

#endif // QUICKBOT_FSM_HPP
