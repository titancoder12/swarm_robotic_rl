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

