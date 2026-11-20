# Weights

Files live in `weights/`, named like the A6000 training runs: `<run_id>-<iter>-<kind>`.
The C++ loader always reads the fixed names in this folder, which are symlinks:

| Slot name (read by C++ / Python) | Points to (active) |
|---|---|
| `base_jit.pt` | `weights/936-19-GO2-QD59-D435I-LR3E4-73000-base_jit.pt` |
| `vision_weight.bin` (EPWT, C++) | `weights/936-19-GO2-QD59-D435I-LR3E4-73000-vision_weight.bin` |
| `vision_weight.pt` (Python) | `weights/936-19-GO2-QD59-D435I-LR3E4-73000-vision_weight.pt` |

## Available checkpoints

| run @ iter | Status | Train setup | Sim result (Isaac, D435i 27.5 deg, motor limit) |
|---|---|---|---|
| `936-19-GO2-QD59-D435I-LR3E4` @ 73000 | **active** | D435i head cam [0.34, 0, 0.07] m, pitch 25.5-29.5, Kp 25 / Kd 0.6, Go2 torque-speed limit | 0.40 m step 27/28 + 15/15 (2 seeds), 0 falls; gaps ~47% |
| `936-19-GO2-QD59-D435I-LR3E4` @ 70500 | alternate | same | 0.40 m ~87%, gaps ~59% |
| `922-10-GO2-STUDENT-D0` @ 59500 | old (robot test 2026-11-05/06) | old cam [0.24, -0.0175, 0.12] m, pitch 21.2-24.6, Kp 40 / Kd 1 | not for the D435i head mount |

Every checkpoint must run with the PD gains it was trained with (`rl_kp` / `rl_kd` in `config.yaml`).

## Switch checkpoint

```bash
cd policy/go2/go2_ep_student
RUN=936-19-GO2-QD59-D435I-LR3E4-70500   # or ...-73000
ln -sfn weights/$RUN-base_jit.pt base_jit.pt
ln -sfn weights/$RUN-vision_weight.bin vision_weight.bin
ln -sfn weights/$RUN-vision_weight.pt vision_weight.pt
```

Rolling back to 922 also needs `rl_kp: 40`, `rl_kd: 1` in `config.yaml`.

## Source

A6000: `~/extreme-parkour/qd59_step40_20261119/deploy/` (`.bin` made with `scripts/export_vision_weight_bin.py`).
