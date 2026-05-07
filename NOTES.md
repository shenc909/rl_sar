# NOTES

Project-specific gotchas and how-to recipes that don't belong in `README.md`
(user-facing) or `CLAUDE.md` (LLM-facing build/architecture map).

## Adding another locomotion policy on go2

The FSM is built around explicit per-policy subclasses of `RLFSMStateRLLocomotion`
(see [src/rl_sar/fsm_robot/fsm_locomotion_common.hpp](src/rl_sar/fsm_robot/fsm_locomotion_common.hpp)
and [src/rl_sar/fsm_robot/fsm_go2.hpp](src/rl_sar/fsm_robot/fsm_go2.hpp)). To add a fourth policy:

1. Drop the trained artefacts at `policy/go2/<your_policy>/{config.yaml, <model>.pt}`.
2. In `fsm_go2.hpp`, declare a new subclass:
   ```cpp
   class RLFSMStateRLLocomotion_YourPolicy : public RLFSMStateRLLocomotion {
   public:
       RLFSMStateRLLocomotion_YourPolicy(RL* rl)
           : RLFSMStateRLLocomotion(rl,
               "RLFSMStateRLLocomotion_YourPolicy",   // FSM state name
               "your_policy") {}                      // policy folder under policy/go2/

       std::string CheckChange() override {
           // List the inter-policy transitions you want to allow from this state.
           if (rl.control.current_keyboard == Input::Keyboard::Num1
            || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
               return "RLFSMStateRLLocomotion_Dreamwaq";
           return RLFSMStateRLLocomotion::CheckChange();
       }

       ChangeDecision CanTransitionTo(const std::string& target) const override {
           if (target == "RLFSMStateRLLocomotion_Dreamwaq")
               return {true, ""};
           return RLFSMStateRLLocomotion::CanTransitionTo(target);
       }
   };
   ```
3. Register it in `Go2FSMFactory::CreateState` and add the state name to
   `Go2FSMFactory::GetSupportedStates()`.
4. Add the hotkey to `RLFSMStateGetUp::CheckChange()` (and the matching
   `CanTransitionTo` widening) so the user can enter the new policy directly
   from the get-up state.
5. (Optional) Add inter-policy transitions *into* this state from the existing
   policies' `CheckChange()` and `CanTransitionTo()` methods.

Compile-time check: omitting any of those edits will produce either an
unreachable hotkey (silent), a service-rejected transition (`success: false,
message: "Transition from '...' to '...' is not in this state's legal targets."`),
or `[FSM] State '...' not found!` at runtime when the user presses the new key.

## Different observation lists per policy

Already supported via `config.yaml`. List the desired tokens under
`observations:` and `RL::ComputeObservation()` (in
[src/rl_sar/library/core/rl_sdk/rl_sdk.cpp](src/rl_sar/library/core/rl_sdk/rl_sdk.cpp))
will assemble the right input on the next inference cycle. The history buffer
is reallocated inside `RL::InitRL()` based on the new config's
`observations_history`, so it is reset on every policy switch (it is *not*
preserved across switches).

If your policy needs an observation token that doesn't already have a handler
in the if-else chain in `ComputeObservation`, add one new `else if` branch.

## Different `dt` / `decimation` per policy

Mechanically supported by the `LoopFunc::SetPeriod` setter and the
`RL::OnConfigSwitched` virtual hook. Each go2 executable's
`OnConfigSwitched()` calls `loop_rl->SetPeriod(dt * decimation)` after every
`SwitchToConfig`, so the policy inference loop retunes itself.

### Pitfalls (read before letting `dt` itself vary)

- **`dt` is a robot-level property, not a policy-level one.** It is the rate at
  which the low-level joint controller / SDK ticks. The robot's onboard kp/kd
  values, motor torque envelopes, and the latency budget of the comms layer
  are all tuned for a particular `dt`. Changing it between policies is a
  controllability hazard.
- **Recommendation:** keep `dt` constant for go2 (use the same value in every
  config under `policy/go2/`). Let `decimation` differ per policy when you need
  a different inference rate — that retunes only `loop_rl`, not `loop_control`,
  and is the safe way to get e.g. a 50 Hz policy alongside a 33 Hz one.
- **If you absolutely must vary `dt`:** un-comment the `loop_control` retune
  inside `RL_Real::OnConfigSwitched` *and* re-validate the kp/kd gains on the
  hardware for each rate. Test in MuJoCo first
  (`./cmake_build/bin/rl_sim_mujoco go2 <SCENE>`) where the cost of getting it
  wrong is just a fall in sim, not a fall on hardware.
- **History buffer is reset every switch.** Policies that depend on a long
  observation history will see one inference cycle of zero-padding (or whatever
  `ObservationBuffer`'s reset fill value is) right after the switch. If that's
  unacceptable, the only fix is to keep two history buffers — out of scope of
  the current architecture.
- **Switch latency is dominated by `InitRL`.** Reloading the model file and
  rebuilding the history buffer takes tens of ms on x86 and longer on the
  Jetson. During that window `RL::Forward` falls back to the last action
  vector. Switching while the robot is unbalanced is unsafe; the existing
  `Num9 → GetDown` exit is your friend.

## Requesting FSM state changes

There are three ways to drive an FSM transition. They all funnel through
`FSM::RequestStateChange`, so the underlying behaviour is identical — the
difference is the input surface.

### 1. Keyboard

While the controller is running, the keys below are polled by `loop_keyboard`
and translated into FSM transition requests. Bindings are defined per state in
[src/rl_sar/fsm_robot/fsm_go2.hpp](src/rl_sar/fsm_robot/fsm_go2.hpp):

| Key | From state | Goes to | Meaning |
|-----|------------|---------|---------|
| `0` | Passive / GetDown | `RLFSMStateGetUp` | Stand up |
| `9` | (most non-Passive) | `RLFSMStateGetDown` | Sit down |
| `P` | (anything except Passive) | `RLFSMStatePassive` | Cut motors / passive mode |
| `1` | GetUp / locomotion | `RLFSMStateRLLocomotion_Dreamwaq` | Default policy |
| `2` | GetUp / locomotion | `RLFSMStateRLLocomotion_RobotLab` | Alternate policy |
| `3` | GetUp / locomotion | `RLFSMStateRLLocomotion_HimLoco` | Alternate policy |

The exact transitions are spelled out in each state's `CheckChange()` —
those are the source of truth. From any locomotion state, the *only* policy
switches that fire are the ones explicitly listed in *that* state's
`CheckChange`; pressing an unmapped key from there is silently ignored.

### 2. Gamepad / wireless controller

Mirror of the keyboard table. Bindings use the chord enums from
`Input::Gamepad` — the existing repo wiring is documented above the enum in
[src/rl_sar/library/core/rl_sdk/rl_sdk.hpp:99-110](src/rl_sar/library/core/rl_sdk/rl_sdk.hpp#L99-L110).
With the convention that R1 + DPad-direction selects a locomotion policy:

| Combo | Goes to | Meaning |
|-------|---------|---------|
| `A` | `RLFSMStateGetUp` | Stand up |
| `B` | `RLFSMStateGetDown` | Sit down |
| `L1+X` | `RLFSMStatePassive` | Passive |
| `R1+DPad-Up` | `RLFSMStateRLLocomotion_Dreamwaq` | Default policy |
| `R1+DPad-Right` | `RLFSMStateRLLocomotion_RobotLab` | Alternate policy |
| `R1+DPad-Down` | `RLFSMStateRLLocomotion_HimLoco` | Alternate policy |

### 3. ROS2 service (for scripts, web UIs, behaviour trees)

Both go2 real-robot binaries (`rl_real_go2` and `rl_real_go2_ros2`) expose:

| Service | Type | Request | Response |
|---------|------|---------|----------|
| `/go2_1/set_fsm_state` | `robot_msgs/srv/SetFsmState` | `string state_name` | `bool success, string message` |

The state name must match one of the registered FSM states exactly:

- `RLFSMStatePassive`
- `RLFSMStateGetUp`
- `RLFSMStateGetDown`
- `RLFSMStateRLLocomotion_Dreamwaq`
- `RLFSMStateRLLocomotion_RobotLab`
- `RLFSMStateRLLocomotion_HimLoco`

Examples:

```bash
# Stand
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateGetUp}"
# Switch to a specific policy
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateRLLocomotion_RobotLab}"
# Sit
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateGetDown}"
# Passive (motors limp; useful as a remote cutoff if you can't reach the keyboard)
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStatePassive}"
# Typo → success: false, message: "Unknown FSM state: '...'"
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateRLLocomotion_RobotLap}"
# Already in that state → success: true, message: "Already in '...'" (no-op)
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateGetDown}"
```

#### Semantics

- `success: true` means the request was *accepted* (the state name is
  registered, the current state's `CanTransitionTo` allowed it, and
  `RequestStateChange` was queued), **not** that the target state's `Enter()`
  has run yet. The transition fires on the FSM's next tick.
- `success: false` happens in two distinct cases, each with a specific reason
  string in the response:
  1. **Unknown state name** — typo or never-registered. `message` reads
     `"Unknown FSM state: '<name>'"`.
  2. **Illegal transition from the current state** — the current state's
     `CanTransitionTo(target)` returned `{false, ...}`. `message` is whatever
     reason that override returned, e.g.
     `"Cannot leave Passive directly to '<X>'; go via GetUp first."` or
     `"GetUp not complete (percent_getup=<x>); transitions to '<X>' not yet allowed."`.

  This means the service mirrors the gating that the keyboard/gamepad path
  already enforces — you cannot use the service to skip the get-up sequence or
  to make an inter-policy jump that the policy itself doesn't allow.
- The service is fire-and-forget after a successful queue. If `Enter()` then
  bounces back to `Passive` (e.g. `SwitchToConfig` throws because the policy
  folder is missing), the service response will *not* reflect that — but the
  asynchronous outcome topic does (see below).

#### Per-state legality table

| Current state | Allowed targets via service |
|---|---|
| `RLFSMStatePassive` | `RLFSMStateGetUp`, self |
| `RLFSMStateGetUp` (ramp complete, `percent_getup >= 1.0`) | `RLFSMStatePassive`, `RLFSMStateGetDown`, any `RLFSMStateRLLocomotion_*`, self |
| `RLFSMStateGetUp` (ramp in progress) | `RLFSMStatePassive`, self — locomotion targets refused with reason |
| `RLFSMStateGetDown` | `RLFSMStatePassive`, `RLFSMStateGetUp`, self |
| `RLFSMStateRLLocomotion_Dreamwaq` | universal exits + `_RobotLab` + `_HimLoco` + self |
| `RLFSMStateRLLocomotion_RobotLab` | universal exits + `_Dreamwaq` + `_HimLoco` + self |
| `RLFSMStateRLLocomotion_HimLoco`  | universal exits + `_Dreamwaq` + `_RobotLab` + self |

The "universal exits" are `RLFSMStatePassive`, `RLFSMStateGetUp`,
`RLFSMStateGetDown`. All locomotion subclasses inherit them from
`RLFSMStateRLLocomotion::CanTransitionTo`.

#### Common workflows

```bash
# Bring up + walk under the default policy
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateGetUp}"
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateRLLocomotion_Dreamwaq}"

# A/B test two policies on the same run
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateRLLocomotion_RobotLab}"
# …drive around, look at logs/CSVs…
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateRLLocomotion_HimLoco}"

# Park the robot and clean up
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStateGetDown}"
ros2 service call /go2_1/set_fsm_state robot_msgs/srv/SetFsmState "{state_name: RLFSMStatePassive}"
```

## Current-state topic (latched)

Both go2 binaries publish on `/go2_1/fsm_state` (`robot_msgs/msg/FsmState`):

```
builtin_interfaces/Time stamp
string current_state
```

The topic is **latched** — published with `keep_last(1) + transient_local +
reliable` QoS — so any subscriber that joins after the executable has started
immediately receives the most recent state without waiting for the next
transition. The publisher fires:

- once at startup, after the FSM's initial state's `Enter()` has run, with
  that state name;
- whenever the FSM completes a transition — keyboard, gamepad, internal
  `RequestStateChange`, or service-driven, all funnel through the same
  callback.

It does **not** fire for:

- service requests that resolve to "already in that state" (no transition occurred);
- service requests rejected by `HasState` or `CanTransitionTo` (no transition occurred).

Subscribers must use compatible QoS. CLI:

```bash
ros2 topic echo --qos-durability transient_local /go2_1/fsm_state
```

In code (rclcpp):

```cpp
rclcpp::QoS sub_qos(rclcpp::KeepLast(1));
sub_qos.transient_local();
auto sub = node->create_subscription<robot_msgs::msg::FsmState>(
    "/go2_1/fsm_state", sub_qos, callback);
```

### Use cases

- **Single source of truth for controller state.** Behaviour trees,
  orchestrators, and dashboards subscribe once and always have the current
  state.
- **Synchronous-style wait after a service request.** Subscribe to
  `/go2_1/fsm_state` *before* calling `/go2_1/set_fsm_state`; on the next
  message, compare `current_state` to your requested state. If it matches, the
  transition succeeded; if it doesn't (most often `RLFSMStatePassive`), the
  transition bounced — typically because `SwitchToConfig` threw inside
  `Enter()` (missing policy folder, bad model file, or joint-mapping
  mismatch). The bounce reason itself isn't carried on the topic; check the
  controller's stdout / log for the underlying error.
- **Audit / replay.** Logging this topic captures the full session timeline of
  state changes regardless of who triggered them.
