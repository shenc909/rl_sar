/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SDK_HPP
#define RL_SDK_HPP

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <exception>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <tbb/concurrent_queue.h>
#include <vector>
#include <memory>
#include <fstream>
#include <mutex>

#include <yaml-cpp/yaml.h>
#include "fsm.hpp"
#include "observation_buffer.hpp"
#include "vector_math.hpp"
#include "inference_runtime.hpp"
#include "logger.hpp"
#include "motion_loader.hpp"
#include "command_shaper.hpp"

template <typename T>
struct RobotCommand
{
    struct MotorCommand
    {
        std::vector<int> mode;
        std::vector<T> q;
        std::vector<T> dq;
        std::vector<T> tau;
        std::vector<T> kp;
        std::vector<T> kd;

        void resize(size_t num_joints)
        {
            mode.resize(num_joints, 0);
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            tau.resize(num_joints, 0.0f);
            kp.resize(num_joints, 0.0f);
            kd.resize(num_joints, 0.0f);
        }
    } motor_command;
};

template <typename T>
struct RobotState
{
    struct IMU
    {
        std::vector<T> quaternion = {1.0f, 0.0f, 0.0f, 0.0f}; // w, x, y, z
        std::vector<T> gyroscope = {0.0f, 0.0f, 0.0f};
        std::vector<T> accelerometer = {0.0f, 0.0f, 0.0f};
    } imu;

    struct MotorState
    {
        std::vector<T> q;
        std::vector<T> dq;
        std::vector<T> ddq;
        std::vector<T> tau_est;
        std::vector<T> cur;

        void resize(size_t num_joints)
        {
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            ddq.resize(num_joints, 0.0f);
            tau_est.resize(num_joints, 0.0f);
            cur.resize(num_joints, 0.0f);
        }
    } motor_state;
};

namespace Input
{
    // Recommend: Num0-GetUp Num9-GetDown N-ToggleNavMode
    //            R-SimReset Enter-SimToggle
    //            M-MotorEnable K-MotorDisable P-MotorPassive
    //            Num1-BaseLocomotion Num2-Num8-Skills(7)
    //            WS-AxisX AD-AxisY QE-AxisYaw Space-AxisClear
    enum class Keyboard
    {
        None = 0,
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        Space, Enter, Escape,
        Up, Down, Left, Right
    };

    // Recommend: A-GetUp B-GetDown X-ToggleNavMode Y-None
    //            RB_Y-SimReset RB_X-SimToggle
    //            LB_A-MotorEnable LB_B-MotorDisable LB_X-MotorPassive
    //            RB_DPadUp-BaseLocomotion RB_DPadOthers/LB_DPadOthers-Skills(7)
    //            LY-AxisX LX-AxisY RX-AxisYaw
    enum class Gamepad
    {
        None = 0,
        A, B, X, Y, LB, RB, LStick, RStick, DPadUp, DPadDown, DPadLeft, DPadRight,
        LB_A, LB_B, LB_X, LB_Y, LB_LStick, LB_RStick, LB_DPadUp, LB_DPadDown, LB_DPadLeft, LB_DPadRight,
        RB_A, RB_B, RB_X, RB_Y, RB_LStick, RB_RStick, RB_DPadUp, RB_DPadDown, RB_DPadLeft, RB_DPadRight,
        LB_RB
    };

    inline const char *GamepadName(Gamepad gamepad)
    {
        static const char *const names[] = {
            "None",
            "A", "B", "X", "Y", "LB", "RB", "LStick", "RStick", "DPadUp", "DPadDown", "DPadLeft", "DPadRight",
            "LB_A", "LB_B", "LB_X", "LB_Y", "LB_LStick", "LB_RStick", "LB_DPadUp", "LB_DPadDown", "LB_DPadLeft", "LB_DPadRight",
            "RB_A", "RB_B", "RB_X", "RB_Y", "RB_LStick", "RB_RStick", "RB_DPadUp", "RB_DPadDown", "RB_DPadLeft", "RB_DPadRight",
            "LB_RB"};
        const auto i = static_cast<size_t>(gamepad);
        return i < sizeof(names) / sizeof(names[0]) ? names[i] : "?";
    }
}

struct Control
{
    Input::Keyboard current_keyboard = Input::Keyboard::None, last_keyboard = Input::Keyboard::None;
    Input::Gamepad current_gamepad = Input::Gamepad::None, last_gamepad = Input::Gamepad::None;

    float x = 0.0f;
    float y = 0.0f;
    float yaw = 0.0f;
    // Held base-height setpoint for policies with a "height_cmd" observation. Unlike x/y/yaw it is not
    // a direct stick mapping: see RL::UpdateHeightCommand / RL::SetHeightCommand.
    float height = 0.0f;
    bool navigation_mode = false;

    void SetKeyboard(Input::Keyboard keyboad)
    {
        if (current_keyboard != keyboad)
        {
            last_keyboard = current_keyboard;
            current_keyboard = keyboad;
        }
    }

    void SetGamepad(Input::Gamepad gamepad)
    {
        if (current_gamepad != gamepad)
        {
            last_gamepad = current_gamepad;
            current_gamepad = gamepad;
        }
    }

    void ClearInput()
    {
        current_keyboard = last_keyboard;
        current_gamepad = Input::Gamepad::None;
    }
};

struct YamlParams
{
    YAML::Node config_node;

    // Get config value by key
    // WARNING: For vectors/containers, store result in a variable before using iterators/references:
    //   ✓ auto vec = params.Get<std::vector<int>>("key"); vec.begin()
    //   ✗ params.Get<std::vector<int>>("key").begin()  // dangling reference!
    template<typename T>
    T Get(const std::string& key, const T& default_value = T()) const
    {
        if (config_node[key])
        {
            return config_node[key].as<T>();
        }
        return default_value;
    }

    bool Has(const std::string& key) const
    {
        return config_node[key].IsDefined();
    }
};

template <typename T>
struct Observations
{
    std::vector<T> lin_vel;
    std::vector<T> ang_vel;
    std::vector<T> gravity_vec;
    std::vector<T> commands;
    std::vector<T> height_cmd;
    std::vector<T> base_quat;
    std::vector<T> dof_pos;
    std::vector<T> dof_vel;
    std::vector<T> actions;
    std::vector<T> height_scan;
};

class RL
{
public:
    RL() {};
    ~RL() {};

    YamlParams params;
    Observations<float> obs;
    std::vector<int> obs_dims;

    RobotState<float> robot_state;
    RobotCommand<float> robot_command;
    tbb::concurrent_queue<std::vector<float>> output_dof_pos_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_vel_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_tau_queue;

    FSM fsm;
    RobotState<float> start_state;
    RobotState<float> now_state;
    bool rl_init_done = false;

    // init
    void InitObservations();
    void InitOutputs();
    void InitControl();
    void InitRL(std::string robot_config_path);
    void InitJointNum(size_t num_joints);
    void SwitchToConfig(const std::string& robot_config_path);
    void SwitchToBase();
    void AutoGenerateJointMapping();

    // Hook fired at the tail of SwitchToConfig so executables can react to a
    // policy swap (e.g. retune control loops if dt/decimation changed).
    // Default no-op; override per-executable as needed.
    virtual void OnConfigSwitched() {}

    // rl functions
    virtual std::vector<float> Forward() = 0;
    std::vector<float> ComputeObservation();
    // Build the height_scan placeholder. Defaults to a uniform 0.31 fill; if the config defines
    // "height_scan_fill" (one constant per channel) the vector is split into "height_scan_num_channels"
    // equal contiguous segments (channel-major), each filled with its constant. Used wherever a robot/sim
    // has no real height-scan source (e.g. rl_sim's flat-ground proxy).
    std::vector<float> MakeHeightScanFill() const;
    virtual void GetState(RobotState<float> *state) = 0;
    virtual void SetCommand(const RobotCommand<float> *command) = 0;
    void StateController(const RobotState<float> *state, RobotCommand<float> *command);
    void ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau);

    // yaml params
    void ReadYaml(const std::string& file_path, const std::string& file_name);

    // csv logger
    std::string csv_filename;
    void CSVInit(std::string robot_name);
    void CSVLogger(const std::vector<float> &torque, const std::vector<float> &tau_est, const std::vector<float> &joint_pos, const std::vector<float> &joint_pos_target, const std::vector<float> &joint_vel);

    // control
    Control control;
    void KeyboardInterface();
    // Height command (only active when the config defines "height_cmd_range"; no-ops otherwise).
    // UpdateHeightCommand rate-integrates a recentering stick axis in [-1, 1] into control.height
    // (full deflection = "height_cmd_rate" units/s, "height_cmd_deadzone" ignored around centre), so the
    // setpoint is held when the stick is released. SetHeightCommand sets it directly (e.g. from a topic).
    // Both clamp to "height_cmd_range".
    void UpdateHeightCommand(float stick, float dt);
    void SetHeightCommand(float value);

    // App joystick: sensor_msgs/Joy on /joy_app from the Clara Control app (via clara_relay).
    // Executables subscribe and forward to AppJoyCallback. SDL GameController layout, the same as
    // joy/game_controller_node and rl_sim's /joy:
    //   buttons: A=0 B=1 X=2 Y=3 Back=4 Guide=5 Start=6 LS=7 RS=8 LB=9 RB=10
    //            DPadUp=11 DPadDown=12 DPadLeft=13 DPadRight=14
    //   axes:    LX=0 LY=1 RX=2 RY=3 LT=4 RT=5   (ROS signs: left/up positive; triggers unused)
    // The app streams at a fixed rate (~50 Hz) as a heartbeat, so buttons are edge-triggered: a
    // gamepad command fires once when the pressed combination changes, not on every message.
    // While messages keep arriving the app owns the sticks, and other stick sources should stand
    // down (check AppJoyActive()). If none arrives for "app_joy_timeout" seconds (default 0.5),
    // AppJoyTick() zeroes the sticks once and hands them back.
    void AppJoyCallback(const std::vector<float> &axes, const std::vector<int> &buttons);
    // Call every control tick before StateController, and after any other source of gamepad commands
    // (e.g. a robot's own remote in GetState), so an app press wins the tick it lands in.
    void AppJoyTick();
    bool AppJoyActive() const { return this->app_joy_active.load(); }
    float AppJoyHeightAxis() const { return this->app_joy_height_axis.load(); }  // RY: rate input for UpdateHeightCommand
    std::atomic<bool> app_joy_active{false};
    std::atomic<float> app_joy_height_axis{0.0f};
    std::atomic<int64_t> app_joy_last_ns{0};
    Input::Gamepad app_joy_last_combo = Input::Gamepad::None;  // AppJoyCallback thread only
    // A new press is handed to the control thread via AppJoyTick rather than set directly: a
    // SetGamepad landing between StateController and ClearInput would be wiped unseen, and with
    // edge triggering there is no repeat to make up for it.
    std::atomic<int> app_joy_pending_combo{static_cast<int>(Input::Gamepad::None)};
    // joystick_scale / app_joy_timeout cached by ReadYaml: AppJoyCallback runs on the ROS executor
    // thread while a policy switch may be rebuilding params.
    std::mutex app_joy_params_mutex;
    std::vector<float> app_joy_scale{1.0f, 1.0f, 1.0f, 1.0f};
    float app_joy_timeout = 0.5f;
    // Joint state for publishing as sensor_msgs/JointState (e.g. for the Clara Control app's 3D view, via
    // clara_relay, or robot_state_publisher). The control thread records it with JointStateTick() right after
    // GetState(); any thread can read it with GetJointStateSnapshot(), in base.yaml joint_names order (the
    // URDF's joint names). Returns false until the first tick or if base.yaml has no joint_names.
    void JointStateTick();
    // Navigation mode (velocity commands from /cmd_vel instead of the sticks) is toggled by X / N on the control
    // thread; other threads (e.g. a set_navigation_mode service) ask for a value with RequestNavigationMode, which
    // NavigationModeTick applies at the start of the next control tick. NavigationMode() is safe from any thread.
    void RequestNavigationMode(bool enabled);
    void NavigationModeTick();  // call at the start of every control tick, before StateController
    bool NavigationMode() const { return this->navigation_mode_snapshot.load(); }
    std::atomic<int> navigation_mode_request{-1};  // -1 none, 0 off, 1 on
    std::atomic<bool> navigation_mode_snapshot{false};
    bool GetJointStateSnapshot(std::vector<std::string> &names, std::vector<double> &position, std::vector<double> &velocity);
    std::mutex joint_state_mutex;  // guards the three vectors below
    std::vector<std::string> joint_state_names;
    std::vector<double> joint_state_position;
    std::vector<double> joint_state_velocity;
    // Velocity command preprocessing between obs.commands and the "commands" observation (see
    // command_shaper.hpp). Configured from the "command_*" / "max_lateral_accel" config keys and reset
    // on every policy load, so each policy entry starts the rate limiter from rest.
    CommandShaper command_shaper;
    void ConfigureCommandShaper();

    // history buffer
    ObservationBuffer history_obs_buf;
    std::vector<float> history_obs;

    // others
    int motiontime = 0;
    std::string robot_name, config_name;
    std::vector<std::string> physical_joint_names;  // from base.yaml; never overwritten by config.yaml
    bool simulation_running = true;
    std::string ang_vel_axis = "body";  // "world" or "body"
    unsigned long long episode_length_buf = 0;
    float motion_length = 0.0;
    int InverseJointMapping(int idx) const;

    // Motion tracking (for mimic/dance tasks)
    std::unique_ptr<MotionLoader> motion_loader;

    // protect func
    void TorqueProtect(const std::vector<float> &origin_output_dof_tau);
    void AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold);

    // rl module
    std::unique_ptr<InferenceRuntime::Model> model;
    // output buffer
    std::vector<float> output_dof_tau;
    std::vector<float> output_dof_pos;
    std::vector<float> output_dof_vel;

    // thread safety
    std::mutex model_mutex;
    // Serializes the inference→push critical section in RunModel against
    // SwitchToConfig. SwitchToConfig holds this for its full duration so that
    // RunModel can't race a stale-ordering ComputeOutput/queue.push between
    // InitRL and the joint-mapping remap. RunModel takes it via try-to-lock and
    // skips its cycle if it can't get it immediately — losing one cycle is
    // harmless (RLControl just keeps the last command).
    std::mutex output_mutex;
};

class RLFSMState : public FSMState
{
public:
    RLFSMState(RL& rl, const std::string& name)
        : FSMState(name), rl(rl), fsm_state(nullptr), fsm_command(nullptr) {}

    RL& rl;
    const RobotState<float> *fsm_state;
    RobotCommand<float> *fsm_command;

    bool Interpolate(
        float& percent,
        const std::vector<float>& start_pos,
        const std::vector<float>& target_pos,
        float duration_seconds,
        const std::string& description = "",
        bool use_fixed_gains = true
    );

    void RLControl();
};

#endif // RL_SDK_HPP
