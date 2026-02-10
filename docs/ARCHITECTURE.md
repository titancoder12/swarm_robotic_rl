# Swarm RL Environment — Architecture and Functionality

This document explains the system design, module responsibilities, data flow, and how each major component works. It is written for engineers who want to extend or audit the environment.

## 1) High-Level Overview

The project implements a multi-agent swarm simulation with:
- A PyGame-based 2D world
- A PettingZoo Parallel API (`reset/step/render/close`)
- Local, partial observations (lidar + local cues)
- A pheromone (stigmergy) field updated by simple rules
- Pluggable locomotion dynamics (tank or hovercraft)
- Independent or shared DQN training

Core idea: agents **do not communicate directly**. They interact only by sensing their local environment, including the pheromone field and nearby agents.

## 2) File Layout

- `env/config.py`
  - Central dataclass (`SwarmConfig`) for all tunable parameters
- `env/swarm_env.py`
  - Environment implementation and simulation logic
- `train/random_rollout.py`
  - Random policy runner for API validation
- `train/independent_dqn_pytorch.py`
  - Minimal DQN training loop with optional shared policy
- `train/sb3_dqn.py`
  - DQN training using Stable-Baselines3 (shared policy)
- `train/rllib_dqn.py`
  - DQN training using RLlib (shared policy)
- `train/train.py`
  - Backend dispatcher for training (custom, SB3, RLlib)
- `train/demo.py`
  - Loads saved checkpoints and renders policy behavior
- `docs/ARCHITECTURE.md`
  - This document

## 3) Environment API (PettingZoo Parallel)

Class: `SwarmEnv`

Methods:
- `reset(seed=None, options=None) -> (obs_dict, info_dict)`
- `step(action_dict) -> (obs_dict, rewards_dict, terminations, truncations, infos)`
- `render(mode="human", fps=60)`
- `close()`

Shapes and types:
- `action_dict`: `dict[str, int]` keyed by agent id (e.g., `agent_0`)
- `obs_dict`: `dict[str, np.ndarray]` each of shape `(obs_dim,)`
- `rewards_dict`: `dict[str, float]`
- `terminations`: `dict[str, bool]` (all targets collected)
- `truncations`: `dict[str, bool]` (time limit)

## 4) World Model

### 4.1 Arena
- 2D bounded rectangle: `width × height`
- Walls are the boundary of the map

### 4.2 Obstacles
- Axis-aligned rectangles, placed randomly
- Agents collide against these (movement rejected)

### 4.3 Targets
- Small circles (food dots)
- Removed when collected by any agent

### 4.4 Agents
Each agent has continuous state:
- Position `(x, y)`
- Heading `theta` (radians)
- Forward velocity `v`
- Yaw rate `omega`
- Lateral velocity `v_lat` (used for hovercraft)

## 5) Action Interface (Joystick Model)

Discrete actions (9 total):
- Throttle: `{-1, 0, +1}`
- Turn: `{-1, 0, +1}`

Actions are mapped into a 3×3 grid of `(throttle, turn)` pairs. These are passed to the active dynamics driver.

This preserves a *sim→real* interface: policy outputs high-level velocity commands, not motor PWM.

## 6) Dynamics Drivers

All drivers implement:

```
apply(agent_state, action, dt, cfg, rng) -> proposed_next_state
```

### 6.1 TankKinematicsDriver
- Differential-drive style kinematics
- Controlled by forward speed and yaw-rate
- Acceleration and angular acceleration are rate-limited

### 6.2 HovercraftDriver
- Same interface as tank
- Adds inertia and lateral drift
- Includes noise and occasional slip for domain randomization

### 6.3 Mode Selection
`SwarmConfig.dynamics_mode`:
- `"tank"`: always tank
- `"hover"`: always hovercraft
- `"mixed"`: randomly choose per episode

## 7) Collision Handling

Collision rules are enforced *after* dynamics propose a new state:
- If the agent would cross walls: move rejected
- If the agent would intersect an obstacle: move rejected
- Collision adds `reward_collision`

## 8) Stigmergy System (Pheromone Field)

Implemented as a 2D grid updated every step.

### 8.1 Deposit Rule
- Each agent deposits `pheromone_deposit` into its current grid cell

### 8.2 Decay Rule
- Field decays every step: `grid *= pheromone_decay`

### 8.3 Diffusion Rule
- Simple neighbor averaging with `pheromone_diffuse_rate`
- Implemented via `np.roll` to avoid costly convolution

### 8.4 Rendering
- Optional heatmap rendering if `render_pheromone` is enabled

## 9) Observations (Local, Partial)

Observation vector per agent is the concatenation of:
1. Lidar rays (normalized distances)
2. Vector to nearest target (agent frame, normalized)
3. Vector to nearest agent (agent frame, normalized)
4. Heading `sin(theta), cos(theta)`
5. Speed (normalized)
6. Pheromone samples (if enabled)

Default `obs_dim` = `9 + 2 + 2 + 2 + 1 + 3 = 19`.

### 9.1 Lidar
- `lidar_rays`: number of rays
- Each ray steps through the environment until it hits a wall or obstacle
- Distances normalized by `lidar_max_range`

### 9.2 Target Cue
- Nearest target selected by Euclidean distance
- Relative vector converted to agent frame

### 9.3 Neighbor Cue
- Nearest agent selected by Euclidean distance
- Relative vector converted to agent frame

### 9.4 Pheromone Cue
- Samples taken along the agent’s forward direction
- Values normalized by local maximum

## 10) Reward Function

Per agent, per step:
- `reward_step`: small negative step cost
- `reward_target`: positive reward on target collection
- `reward_collision`: negative reward on collision

Episode ends when:
- All targets collected (`terminated = True`) OR
- `max_steps` reached (`truncated = True`)

## 11) Training Architecture (DQN)

`train/independent_dqn_pytorch.py` implements:
- Separate replay buffer per agent
- Independent Q-networks by default
- Optional shared-policy mode (`--shared-policy`)

Optional library backends:
- SB3 DQN: `train/sb3_dqn.py` (shared policy with SuperSuit vectorization)
- RLlib DQN: `train/rllib_dqn.py` (shared policy via RLlib multi-agent config, with compatibility shims for current Ray APIs)

### 11.1 Replay Buffer
Stores tuples:
```
(obs_i, action_i, reward_i, next_obs_i, done)
```

### 11.2 Network
Simple MLP:
- `obs_dim → 128 → 128 → action_dim`

### 11.3 Exploration
- Epsilon-greedy with linear decay
- Per-step epsilon is global; each agent samples independently

### 11.4 Target Network
- Updated every `target_update` steps

### 11.5 Checkpointing
- Independent: `checkpoints/agent_{i}.pt`
- Shared: `checkpoints/shared.pt`

## 12) Demo Runtime

`train/demo.py` loads saved checkpoints and runs inference in the live simulation.

## 13) Extensibility Points

Common extension points:
- Swap dynamics by editing or adding a driver
- Add new observation channels (e.g., nest direction)
- Add new rewards (e.g., return-to-nest)
- Multi-pheromone fields (food vs nest)
- Curriculum by adjusting `n_targets`, `n_obstacles`, `max_steps`

## 14) Known Limitations (MVP)

- No carrying state (food transport) yet
- No explicit nest behavior
- Pheromone is a single type
- Simple collision model (axis-aligned rectangles)

## 15) Suggested Next Steps

If you want to evolve this toward a full research environment:
- Add carry state + nest reward
- Add two pheromone channels
- Add boundary conditions for pheromone diffusion
- Add observation noise or domain randomization
- Add environment wrappers for vectorized training

## 16) How to Modify the Code

This section gives practical guidance for common changes. Most tweaks are done in `env/config.py`.

### Change world size, counts, or episode length
- Edit `SwarmConfig` in `env/config.py`:
  - `width`, `height`
  - `n_agents`, `n_targets`, `n_obstacles`
  - `max_steps`

### Change rewards
- Edit `reward_target`, `reward_step`, `reward_collision` in `env/config.py`.
- If you need new reward terms, modify `SwarmEnv.step()` in `env/swarm_env.py`.

### Change observations
- Lidar: adjust `lidar_rays`, `lidar_max_range`, `lidar_step` in `env/config.py`.
- Pheromone cue: toggle `obs_include_pheromone` or change `pheromone_samples`.
- To add new channels, edit `_get_obs()` and update the concatenation order.

### Change actions or dynamics
- Actions are currently discrete (9 actions). The mapping lives in `_build_action_table()` in `env/swarm_env.py`.
- Dynamics live in `TankKinematicsDriver` and `HovercraftDriver`.
- To add a new driver, create a new class that implements `apply(...)` and update `_select_driver()`.

### Change pheromone behavior
- Deposit/decay/diffuse parameters live in `env/config.py`:
  - `pheromone_deposit`, `pheromone_decay`, `pheromone_diffuse_rate`
- The update logic lives in `_update_pheromone()` in `env/swarm_env.py`.

### Add new targets or tasks
- Targets are spawned in `_spawn_targets()` and collected in `_handle_targets()` in `env/swarm_env.py`.
- For nest-return tasks, add a nest region and update rewards/termination logic in `step()`.

### Training changes
- DQN architecture: edit `QNetwork` in `train/independent_dqn_pytorch.py`.
- Exploration schedule: edit `linear_schedule()` and DQN config values.
- To train shared policy, pass `--shared-policy` to the training script.

### Keep configs and code in sync
- If you add new observation channels or actions, update:
  - `obs_dim` assumptions in training scripts
  - README documentation for actions/observations
