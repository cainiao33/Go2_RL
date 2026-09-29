# Go2 Quadruped Reinforcement Learning: Autonomous Navigation + Locomotion

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/🇨🇳_语言-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README_en.md"><img src="https://img.shields.io/badge/🇬🇧_Language-English-blue?style=for-the-badge" alt="English"></a>
</p>

> **One-liner**: training a Unitree Go2 (12 actuated joints) with PPO (reward engineering deeply aligned with the scoring formula + staged curriculum learning) to walk stably over slopes, stairs, mazes and race tracks, and navigate autonomously to goal points.
> **Results**: 2026 Tencent AI Arena Global Open Competition D02 — **2nd** in the Central Regional Preliminary · 5th in the Regional Final · **Second Prize** in the National Final.
> **How to run**: training must happen inside the **Tencent AI Arena RL platform** (application required; Isaac Lab simulation and the KaiwuDRL framework are provided platform-side and cannot run locally). Fellow platform entrants: `python train_test.py` (Standard / Track mode switching via `Config.CURRENT` in `agent_ppo/conf/conf.py`); full workflow in [docs/TRAINING.md](docs/TRAINING.md) (Chinese).

---

## ⚠️ Reproduction Notes

This repository **does not provide step-by-step off-platform reproduction**, because training and evaluation must take place inside the **Tencent AI Arena (Kaiwu) RL platform** (<https://aiarena.tencent.com/>) — the platform is **application-based**: you must register, apply, and pass review before use —

- Simulation is based on Isaac Lab, with environment code injected platform-side (`isaac_env/` in this repo is only a placeholder directory)
- Training depends on platform modules such as `kaiwudrl` / `common_python` / `tools.*`, which are absent from local environments
- Model evaluation (`agent.exploit()`) and official submissions are done through platform tasks

What can be verified locally has been captured as static tests (no platform dependencies): `pytest tests` (observation-layout ABI / agent_diy slot contract / config parsing / ActorCritic forward; torch cases auto-skip where torch is absent). CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

The repository is therefore positioned as a **complete open-sourcing of the solution itself**: reward engineering, network architecture, observation design, curriculum learning and tuning details are all visible. Fellow platform entrants can import the code package and train following the platform workflow; other readers can use it as an RL engineering reference for quadruped locomotion + navigation.

---

## The Competition

Train an RL agent to control a **Unitree Go2 quadruped robot** (12 actuated joints) for autonomous navigation and locomotion in simulation, walking stably over complex terrain (slopes, stairs, mazes, race tracks) and reaching goal points quickly.

**Two competition modes**:
- **Standard mode**: walk over multiple complex terrains; composite score (distance 40% + time 20% + energy 20% + posture 20%)
- **Track mode**: navigate from start to finish along concatenated race tracks; scored on completion rate, time, posture and energy

---

## Core Algorithm

**PPO (Proximal Policy Optimization)** + deep reward engineering + staged curriculum learning

```
Actor:  obs[301] → [512, 256, 128] → actions[12]
Critic: critic_obs[316] → [512, 256, 128] → value[1]
```

---

## Six Technical Highlights

### 1. Reward engineering deeply aligned with the scoring formula

Every reward weight was reverse-engineered from the scoring formula `0.4×distance + 0.2×time + 0.2×energy + 0.2×posture`, keeping the training target identical to the competition score. **14+ reward terms** across six layers: base locomotion drive, safety constraints, energy efficiency, gait quality, stair-descent specials, and navigation guidance.

### 2. Adaptive posture penalty (terrain-aware)

A plain `flat_orientation` penalizes equally on all terrain, making the robot struggle on slopes/stairs. This project adapts the penalty weight from `projected_gravity`: strict on flat ground (weight 1.0), lenient on slopes (0.2). **Stair-descent pass rate improved from ~40% to ~75%** (an observational estimate from platform monitors during training; training logs are not archived in the repo — for reference).

### 3. Navigation command-injection layer (Track mode)

Key decision: **do not retrain the whole model** — navigation becomes a velocity-command generation layer. The locomotion policy trained in the Standard stage stays intact; the navigation layer only produces goal-directed `[vx, vy, wz]` commands. Benefits: locomotion ability preserved, fast transfer, model-compatible.

### 4. Three-layer NaN/Inf numerical protection

```
Layer 1: loss check    → skip the mini-batch if non-finite
Layer 2: grad check    → zero grads and skip the step on NaN
Layer 3: std clamping  → replace NaN/Inf, clamp to [min_std, 1e6]
```

No numerical collapse occurred throughout the competition training (an observational conclusion from the training period; logs are not archived in the repo).

### 5. Stair-descent specific optimization

For the weak terrain `pyramid_stairs_inv`, two dedicated rewards were designed — `stairs_descend_progress` (rewarding forward speed while descending) and `adaptive_orientation` (allowing moderate tilt) — targeting the fall-prone stair-descent problem.

### 6. Obstacle-avoidance scanner integration (Track mode, off by default)

Obstacle avoidance using the platform's forward-looking `nav_scanner`: slow down and turn when an obstacle is < 0.55 m ahead; force a turn when a dead end is detected (both sides < 0.75 m); dynamically adjust velocity commands to avoid collision termination. Honest note: the pipeline is fully implemented and integrated into the command-injection layer, but it is **disabled in the final submitted configuration** (`nav_cmd_enable_scanner_avoidance = False` in `agent_ppo/conf/conf.py`; kept off out of caution). Enabling it is a one-line change.

---

## Technical Documentation (Chinese)

| Doc | Contents |
|------|------|
| [📄 Technical overview](docs/TECH_OVERVIEW.md) | solution selection, staged training route, model architecture |
| [📄 Reward engineering](docs/REWARD_ENGINEERING.md) | 14+ reward terms, weight derivation, champion-solution transfer |
| [📄 Navigation design](docs/NAVIGATION.md) | command-injection layer, obstacle avoidance, Track-mode implementation |
| [📄 Training strategy & curriculum](docs/TRAINING.md) | domain randomization, velocity-command curriculum, hyperparameter schedule, monitoring |
| [📄 Code structure](docs/CODE_STRUCTURE.md) | directory layout and module responsibilities |

---

## Quick Start

```bash
# Training entry (platform-only; see Reproduction Notes above; train_test.py defaults to algorithm_name = "ppo")
python train_test.py

# Switch training stage (Standard / Track)
# edit Config.CURRENT in agent_ppo/conf/conf.py (the only effective place)

# Local static checks (no platform dependencies)
pytest tests
```

---

## Provenance and Licensing

This repository is built on the **Tencent AI Arena (Kaiwu) competition code package** (`kaiwu.json` declares `project_code = legged_robot_competition_26`):

- Template files carrying the `Copyright © 1998 - 2026 Tencent. All Rights Reserved` header, and the platform documents under `docs/` (development guide / framework docs / distributed computing framework, etc.) are copyrighted by **Tencent**, used under the platform and competition terms
- The original parts of this repository — `agent_ppo/feature/nav_command.py`, `nav_signal.py`, `agent_ppo/tool/scan.py`, `docs/适配方案/`, the 5 technical documents at the `docs/` root, `README.md`, `AGENTS.md`, `agent_diy/README.md`, `tests/` and tooling config (`.github/workflows/ci.yml`, `pyproject.toml`) — are created by cainiao33 and released under **MIT**
- Files deeply modified on top of Tencent templates (`track_tensor_bridge.py`, `feature_layout.py`, etc.) keep the Tencent header; the modifications are likewise © cainiao33 (MIT)
- Accordingly, this repository **does not provide a single whole-repo LICENSE** (to avoid re-licensing the Tencent "All Rights Reserved" templates along with it)

## AI Collaboration Note

Reward-engineering tuning and platform training were done by the author; repository engineering (static tests, CI, agent_diy dead-code cleanup, doc corrections and restructuring) was done with **Claude Code** assistance — see commit history.

---

## Competition Results

| Stage | Rank | Notes |
|------|------|------|
| **Central Regional Preliminary** | **2nd** | Standard + Track combined |
| **Regional Final** | **5th** | qualifiers from all regions |
| **National Final** | **Second Prize** | final national ranking |

---

> **Author**: cainiao33
>
> **Repository**: https://github.com/cainiao33/Go2_RL
>
> **Platform**: [Tencent AI Arena (Kaiwu)](https://aiarena.tencent.com/) (application required)
