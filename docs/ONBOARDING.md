# Developer Ramp-Up: RL + Code Walkthrough

This guide maps RL concepts directly to the code in this repo. It is meant to help new contributors understand how the environment, training, and demos fit together.

## 1) Environment = The MDP (`env/swarm_env.py`)

`SwarmEnv` defines the RL world:
- **`reset()`** returns initial observations.
- **`step(action_dict)`** advances the world and returns `(obs_dict, rewards_dict, terminations, truncations, infos)`.
- **`render()`** draws the environment (not part of RL math, but useful for inspection).

### Actions
`_build_action_table()` maps 9 discrete actions to `(throttle, turn)`:
- Action space: `Discrete(9)`
- Joystick-style control

### Observations
`_get_obs()` builds a per-agent vector of length 19:
- Lidar rays
- Nearest target vector
- Nearest neighbor vector
- Heading (sin, cos)
- Speed
- Pheromone samples

### Rewards
Defined in `step()` and `_handle_targets()`:
- `reward_step`: small negative step cost
- `reward_collision`: penalty for collisions
- `reward_target`: reward for collecting targets

### Episode End
`terminated`: all targets collected  
`truncated`: max steps reached

### API
This env is PettingZoo Parallel API:
- Dict-based obs/actions for each agent.


## 2) Custom DQN Trainer (`train/independent_dqn_pytorch.py`)

This is a minimal DQN implementation for multi-agent training.

### Replay Buffer
`ReplayBuffer` stores transitions `(s, a, r, s', done)` and returns random minibatches.

### Q-Network
`QNetwork` maps observations to Q-values for each discrete action.

### Training Loop
Key steps:
1. `reset()` environment
2. Epsilon-greedy action selection
3. `step()` environment
4. Store transitions in replay buffer
5. Sample minibatch and apply Bellman update
6. Periodically sync target network

### Multi-Agent Modes
- **Independent**: one Q-network per agent
- **Shared**: one Q-network shared by all agents


## 3) Demos (`train/demo.py`)

`demo.py` runs trained policies and renders the environment.

Backends:
- **custom**: loads `.pt` checkpoints (your own DQN)
- **sb3**: loads Stable-Baselines3 DQN `.zip`
- **rllib**: loads RLlib checkpoint directory

Supports `--max-steps` to auto-exit after N steps (useful for smoke tests).


## 4) Training Dispatcher (`train/train.py`)

`train/train.py` selects a backend with `--backend`:
- `custom`: uses your own DQN
- `sb3`: Stable-Baselines3 DQN
- `rllib`: RLlib DQN


## 5) RL Backends (SB3 / RLlib)

### SB3
`train/sb3_dqn.py` wraps the PettingZoo env with SuperSuit and trains DQN.

### RLlib
`train/rllib_dqn.py` uses RLlib’s DQN with compatibility shims for current Ray versions.  
If Ray warns about `/tmp` or socket length, use:
```
--ray-tmpdir /Users/christopherlin/.ray_tmp
```


## 6) Common Commands

Custom training (headless):
```
python train/train.py --backend custom --headless --total-steps 10000
```

SB3 training:
```
python train/train.py --backend sb3 --headless --total-steps 10000 --save-path checkpoints/sb3_dqn.zip
```

RLlib training:
```
python train/train.py --backend rllib --headless --total-steps 10000 --save-dir checkpoints/rllib_dqn --ray-tmpdir /Users/christopherlin/.ray_tmp
```

Custom demo:
```
python train/demo.py --backend custom --checkpoint-dir checkpoints
```

SB3 demo:
```
python train/demo.py --backend sb3 --sb3-model checkpoints/sb3_dqn.zip
```

RLlib demo:
```
python train/demo.py --backend rllib --rllib-checkpoint checkpoints/rllib_dqn --ray-tmpdir /Users/christopherlin/.ray_tmp
```

