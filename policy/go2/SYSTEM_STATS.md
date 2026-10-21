# Go2 onboard computer: system stats

Snapshot taken 2026-10-21 on the Go2's Jetson (`hostname: ubuntu`, user `unitree`),
while checking whether `loop_rl` (50 Hz) and `loop_control` (200 Hz) can hold their rates.

## Board / OS

| Item | Value |
|---|---|
| L4T release | R35 (release), REVISION 3.1, BOARD t186ref, built 2023-03-19 |
| Kernel | `5.10.104-tegra #1 SMP PREEMPT Sun Mar 19 07:55:28 PDT 2023 aarch64` |
| Kernel cmdline | `root=PARTUUID=19274313-2e65-42a1-b5e1-6a2ec7a1744b rw rootwait rootfstype=ext4 mminit_loglevel=4 console=ttyTCU0,115200 console=ttyAMA0,115200 firmware_class.path=/etc/firmware fbcon=map:0 net.ifnames=0 nv-auto-config` |
| nvpmodel mode | **MAXN** (mode 0), config `/etc/nvpmodel.conf` |
| Module | **Jetson Orin NX**: Cortex-A78 CPU, ga10b GPU, 2x DLA, PVA |

## CPU

| Item | Value |
|---|---|
| Architecture | aarch64, Cortex-A78 (nvpmodel `CPU_A78_*`); lscpu reports "ARMv8 Processor rev 1 (v8l)", r0p1 |
| Cores | 8 online (0-7), 1 thread/core, 2 clusters x 4 cores: cpu0-3 (`CPU_A78_0`), cpu4-7 (`CPU_A78_1`) |
| Clock (hardware range, lscpu) | 115.2 MHz to 1984 MHz |
| Clock (current scaling limits, both clusters) | `scaling_min_freq` 729.6 MHz, `scaling_max_freq` 1984 MHz |
| Cache | L1d 512 KiB, L1i 512 KiB, L2 2 MiB, L3 4 MiB (totals) |
| cpufreq governor | `schedutil` on all 8 cores |

The board is already in MAXN, but the clocks are not locked: `schedutil` moves each
cluster between 729.6 and 1984 MHz. `sudo jetson_clocks` would pin them at the maximum.

## GPU and accelerators (nvpmodel MAXN, current values)

| Item | Value |
|---|---|
| GPU (ga10b) frequency | 306 MHz min, 918 MHz max (devfreq) |
| GPU power control | `auto` (runtime power management on) |
| GPU gating masks | `FBP_PG_MASK` 2, `TPC_PG_MASK` 240 |
| EMC (memory) cap | 0 (uncapped) |
| DLA0 / DLA1 core cap | 614.4 MHz |
| DLA0 / DLA1 falcon cap | 294.4 MHz |
| PVA0 VPS / AXI cap | 704 MHz / 486.4 MHz |

## Scheduling / real-time

| Item | Value | Implication |
|---|---|---|
| Preemption model | `CONFIG_PREEMPT=y` (not PREEMPT_RT) | `SCHED_FIFO` latency is well under the 20 ms budget |
| Tick rate | `CONFIG_HZ=250` (4 ms tick) | |
| Tickless | `CONFIG_NO_HZ_IDLE=y`, `CONFIG_NO_HZ_FULL` **not set** | `isolcpus` alone cannot stop the tick on an isolated core |
| Isolated CPUs | none (`/sys/devices/system/cpu/isolated` empty) | |
| `ulimit -r` (RT priority limit) | `0` | `SCHED_FIFO` fails until `rtprio` is raised in `/etc/security/limits.d/` or `CAP_SYS_NICE` is granted |

## Processes running at the time

- Several `python3` processes, all `SCHED_OTHER` (TS), nice 0. One of them (PID 2173)
  had more than 15 threads spread across all 8 cores.
- No `rl_real_go2*` process was running.

## rl_sar settings relevant to timing

- `policy/go2/base.yaml`: `dt: 0.005`, `decimation: 4`, so `loop_control` targets 200 Hz and `loop_rl` targets 50 Hz.
- Inference is single-threaded: `torch::set_num_threads(1)` / ONNX `SetIntraOpNumThreads(1)`
  in `src/rl_sar/library/core/inference_runtime/inference_runtime.cpp`.
- No `SCHED_FIFO`, `mlockall` or CPU pinning is used: `LoopFunc` supports `bindCPU`, but no loop sets it.

## Commands used

```bash
nproc; lscpu
cat /proc/cmdline
uname -a
zcat /proc/config.gz | grep -E "PREEMPT|HZ=|NO_HZ"
cat /sys/devices/system/cpu/isolated
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
ulimit -r
cat /etc/nv_tegra_release
ps -eLo pid,tid,cls,rtprio,ni,psr,comm
sudo nvpmodel -q --verbose
```
