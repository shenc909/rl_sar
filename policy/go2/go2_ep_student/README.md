# go2_ep_student

Extreme Parkour Go2 Student policy slot for `rest271828/rl_sar`.
Active checkpoint: `936-19-GO2-QD59-D435I-LR3E4` @ 73000 (D435i head mount, Kp 25 / Kd 0.6). See `WEIGHTS.md`.
The sections below were written for the older `922-10` / `model_59500` and are kept for reference.

## Real robot (936 @ 73000)

1. Camera: D435i depth 848x480 @ >= 30 Hz on the Jetson; trained mount pitch 25.5-29.5 deg.
   The 2026-11-06 floor depth logs fit to ~35-37 deg camera-to-ground pitch at stand: measure before testing.
2. Jetson: `python3 -u ~/jetson_depth_tcp_bridge.py --bind 0.0.0.0 --port 9876` (see script header for ROS env).
3. Laptop: `RL_SAR_GO2_CONFIG=go2_ep_student RL_SAR_EP_DEPTH_HOST=192.168.123.18 ./cmake_build/bin/rl_real_go2 <iface>`
   (no rebuild needed for a weight swap; `RL_SAR_DRY_RUN=1` first to check depth/obs without sending lowcmd).
4. Gamepad: A = stand up, R1+Up = policy, B = get down, L1+X = passive (e-stop).
5. Command: left stick forward. `cmd_x = ly`, so full forward = 1.0 m/s (trained 0.8-1.2). Start ~2-2.5 m in front of the
   platform, facing it (sim2sim used 2.5 m). Stop the policy (B) after the climb: there are no waypoints past the course, heading drifts.

Does **not** replace `himloco` / `robot_lab`.

## Artifacts

| File | Role |
|---|---|
| `base_jit.pt` | TorchScript `(obs[753], depth_latent[32]) -> action[12]` (estimator inside) |
| `vision_weight.pt` | RecurrentDepthBackbone weights (Python / Isaac) |
| `config.yaml` | rl_sar `InitRL` + EP scales / `joint_mapping` |

See `WEIGHTS.md` for symlink targets.

## Depth cadence (`update_interval=5`)

**错误节拍** = 把 depth GRU 当成 50 Hz 每步都推，而环境只每 5 个控制步才给新深度帧。正确做法：仅当 `extras["depth"] is not None` 时前进 encoder/GRU，其余步复用 latent/yaw。

Enforced in: `sim_isaac_bridge.py`, `infer_offline.py`. MuJoCo has no camera → latent stays 0 (no GRU).

## Isaac Gym MXD (parkour, 1500×192)

| Backend | MXD (waypoints/7) | Notes |
|---|---:|---|
| `jit` (deploy) | **0.90** | vision_weight + base_jit; estimator; no GT priv |
| `ckpt_est` | **0.90** | depth_actor + estimator (fair twin of jit) |
| `ckpt` | **0.88** | evaluate.py oracle with GT priv_explicit |

```bash
source /home/yihan/extreme-parkour/activate.sh
cd /home/yihan/extreme-parkour/rest271828-rl_sar
python policy/go2/go2_ep_student/sim_isaac_bridge.py --backend jit --terrain parkour --steps 1500 --num-envs 192
```

Critical deploy detail: depth backbone output activation must be **ELU** (training default), not Tanh.

## MuJoCo (`rl_sim_mujoco`)

```bash
RL_SAR_GO2_CONFIG=go2_ep_student RL_SAR_AUTO_RL=1 RL_SAR_CMD_X=0 \
  bash scripts/run_go2_mujoco_xvfb.sh
# or
bash scripts/smoke_go2_ep_mujoco.sh
```

- Dual-input Forward + 753-d EP obs (proprio history, `joint_mapping` FL-first, parkour_flat one-hot)
- `RL_SAR_AUTO_RL=1` locks cmds (blocks Xvfb phantom keys)
- **No depth camera** → zero latent; parkour student is unstable here (not a vision MXD score)

## Limits

- MuJoCo cannot score vision parkour without a depth camera / encoder path
- Gazebo/`rl_sim` still needs ROS
