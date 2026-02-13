# Q&A

This file captures recurring questions and answers discussed during development so future contributors can reference decisions quickly.

## Q: Is `ReplayBuffer` the same as a trajectory?
A: No. The replay buffer stores **individual transitions** `(s, a, r, s', done)` and does not preserve episode order. A trajectory is an **ordered sequence** of transitions. The buffer may contain pieces of trajectories, but it is not itself a trajectory.

## Q: Does PettingZoo replace PyTorch or do learning/inference?
A: No. PettingZoo is only the **environment API**. Learning/inference is done by an RL algorithm (custom DQN, SB3, RLlib) implemented separately, typically using a deep learning framework like PyTorch.

## Q: If I switch to an RL library, does the environment need to change?
A: Usually no. As long as `SwarmEnv` follows the PettingZoo Parallel API, you can swap the learning backend without changing the environment. You may add wrappers if a library expects a different API.

## Q: Is inference the same across RL libraries?
A: No. Each library saves models differently and has its own load/predict API. A unified demo script can route to the correct loader based on `--backend`.

## Q: Is `swarm_env.py` “RL code”?
A: It is the **environment** that defines the MDP: observations, actions, rewards, and termination. It does not learn. The training code is separate.

## Q: In sim-to-real transfer, is `_get_obs()` the key interface?
A: Largely yes: `_get_obs()` maps world state to the observation vector the policy uses. On a robot, the equivalent is the **sensor preprocessing pipeline** that produces the same vector. You also need an **action interface** that maps actions to motor commands.

## Q: Does `_handle_targets` reward agents based on distance, and how would this work in the real world?
A: In the sim, `_handle_targets()` gives a reward **only when a target is collected** (within the target radius). It does **not** reward based on distance. In the real world, you’d need a **detectable target** (e.g., marker/vision/RFID) so the robot can confirm contact, or you’d change the task so rewards come from its own sensors (e.g., signal strength/proximity). The observation pipeline must be built from real sensors, and “target collected” must be detectable by the robot.

## Q: If reward depends on a target radius, does the environment need to know target locations?
A: In simulation, yes—the environment tracks target positions to determine if an agent is within the collection radius. That’s not necessarily distance-shaped reward; it’s a **binary “collected” check**. In the real world, the equivalent is a **target detection signal** from sensors, not a hidden oracle.

## Q: What needs to change for real-world deployment?
A: See `docs/ToDo.md` for a required-change checklist covering sensors/observations, action interface, target detection, reset procedures, safety, domain gap mitigation, runtime loop, and logging.

## Q: What does it mean that the env is PettingZoo-native?
A: It means `SwarmEnv` implements the **PettingZoo Parallel API** directly (dict-based `reset()` and `step()` with per-agent observations/rewards/terminations). You can plug it into PettingZoo-compatible tooling without extra wrappers.

## Q: What does MLP stand for?
A: **Multi-Layer Perceptron** — a standard feedforward neural network with fully connected layers.

## Q: Do SB3/RLlib do the training themselves or use an underlying DL library?
A: They **do the training**, but they are built on top of a deep learning framework. SB3 uses **PyTorch**. RLlib supports **PyTorch** (and sometimes other backends). The library manages the RL algorithm loop; the DL framework handles neural nets and gradients.

## Q: What’s the relationship between PettingZoo, SB3/RLlib, and PyTorch?
A: PettingZoo is the **environment API** (multi-agent interface). SB3/RLlib are **RL training libraries** that implement algorithms and use environments. PyTorch is the **deep learning backend** used by those libraries to build and train neural networks. In short: PettingZoo (env) → SB3/RLlib (RL algorithm) → PyTorch (NNs/gradients).
