"""
Classic DQN / Q-learning setup (with replay buffer + target network)
and an ε-greedy behavior policy.
"""
from __future__ import annotations

import argparse  # CLI argument parsing.
import os  # Filesystem paths.
import random  # Epsilon-greedy randomness.
import sys  # Path tweaks for local imports.
from dataclasses import dataclass  # Simple config container.
from typing import List  # Type hints.

import numpy as np  # Numeric arrays for buffers/obs.
import torch  # Neural nets + tensors.
import torch.nn as nn  # NN modules.
import torch.optim as optim  # Optimizers (Adam).

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Repo root.
if ROOT not in sys.path:  # Ensure local imports work.
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig  # Environment config.
from env.swarm_env import SwarmEnv  # PettingZoo env.


@dataclass  # Declarative hyperparameter container.
class DQNConfig:
    """Hyperparameters for DQN training."""
    gamma: float = 0.98 # Discount factor for future rewards.
    batch_size: int = 64 # Minibatch size for sampling from replay buffer.
    buffer_size: int = 50_000 # Experience memory size.
    lr: float = 3e-4 # Learning rate for Adam optimizer.
    target_update: int = 200
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 8000
    warmup_steps: int = 500


class ReplayBuffer:
    """Simple FIFO replay buffer for experience tuples."""
    def __init__(self, capacity: int, obs_dim: int):
        """Allocate storage for fixed-size experience arrays."""
        self.capacity = capacity  # Max number of stored transitions.
        self.obs_dim = obs_dim  # Observation vector length.
        self.ptr = 0  # Write pointer (circular buffer).
        self.size = 0  # Current number of stored transitions.
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s_t
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s_{t+1}
        self.actions = np.zeros((capacity,), dtype=np.int64)  # a_t
        self.rewards = np.zeros((capacity,), dtype=np.float32)  # r_t
        self.dones = np.zeros((capacity,), dtype=np.float32)  # done flag

    def add(self, obs, action, reward, next_obs, done):
        """Insert a transition, overwriting oldest when capacity is exceeded."""
        self.obs[self.ptr] = obs  # Store state.
        self.actions[self.ptr] = action  # Store action.
        self.rewards[self.ptr] = reward  # Store reward.
        self.next_obs[self.ptr] = next_obs  # Store next state.
        self.dones[self.ptr] = done  # Store done.
        self.ptr = (self.ptr + 1) % self.capacity  # Advance pointer (wrap).
        self.size = min(self.size + 1, self.capacity)  # Track current size.

    def sample(self, batch_size: int):
        """Sample a random minibatch and return as torch tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)  # Random indices.
        return (
            torch.tensor(self.obs[idx]),  # Batch of states.
            torch.tensor(self.actions[idx]),  # Batch of actions.
            torch.tensor(self.rewards[idx]),  # Batch of rewards.
            torch.tensor(self.next_obs[idx]),  # Batch of next states.
            torch.tensor(self.dones[idx]),  # Batch of done flags.
        )


def _dict_to_array(data, agent_ids, dtype=None):
    arr = np.array([data[agent] for agent in agent_ids])
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _array_to_dict(arr, agent_ids):
    return {agent: arr[i] for i, agent in enumerate(agent_ids)}


class QNetwork(nn.Module):
    """Small MLP mapping observations to Q-values for each discrete action."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # Two hidden layers with ReLU nonlinearity.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),  # Input -> hidden.
            nn.ReLU(),  # Nonlinearity.
            nn.Linear(128, 128),  # Hidden -> hidden.
            nn.ReLU(),  # Nonlinearity.
            nn.Linear(128, action_dim),  # Hidden -> Q-values for each action.
        )

    def forward(self, x):
        """Return Q-values for each action."""
        return self.net(x)  # Forward pass.


def linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    """Linearly anneal a value from start to end over decay_steps."""
    if step >= decay_steps:  # Stop decaying after schedule ends.
        return end
    frac = step / decay_steps  # Progress ratio.
    return start + frac * (end - start)  # Linear interpolation.


def train(args):
    """Train independent (or shared) DQN policies for each agent."""
    # 1) Environment and config setup.
    cfg = SwarmConfig(n_agents=args.n_agents)  # Env config.
    dqn_cfg = DQNConfig()  # Training config.

    env = SwarmEnv(cfg, headless=args.headless)  # Create env.
    obs_dict, _ = env.reset(seed=args.seed)  # Reset -> initial observations.
    agent_ids = env.possible_agents  # Ordered agent IDs.
    obs = _dict_to_array(obs_dict, agent_ids, dtype=np.float32)  # Dict -> array.
    obs_dim = obs.shape[1]  # Observation length.
    action_dim = cfg.num_actions  # Number of discrete actions.

    # Use GPU if requested and available.
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")  # Device.

    # 2) Initialize Q-networks, target networks, optimizers, and replay buffers.
    if args.shared_policy:
        # One shared policy across all agents.
        q_net = QNetwork(obs_dim, action_dim).to(device)  # Online Q-network.
        target_net = QNetwork(obs_dim, action_dim).to(device)  # Target Q-network.
        target_net.load_state_dict(q_net.state_dict())  # Sync target.
        optimizer = optim.Adam(q_net.parameters(), lr=dqn_cfg.lr)  # Optimizer.
        buffers = [ReplayBuffer(dqn_cfg.buffer_size, obs_dim) for _ in range(cfg.n_agents)]  # Per-agent replay.
        q_nets = [q_net for _ in range(cfg.n_agents)]  # Shared network reference.
        target_nets = [target_net for _ in range(cfg.n_agents)]  # Shared target reference.
        optimizers = [optimizer for _ in range(cfg.n_agents)]  # Shared optimizer reference.
    else:
        # Independent policies (one per agent).
        q_nets = []  # One Q-net per agent.
        target_nets = []  # One target net per agent.
        optimizers = []  # One optimizer per agent.
        buffers = []  # One replay buffer per agent.
        for _ in range(cfg.n_agents):
            net = QNetwork(obs_dim, action_dim).to(device)  # Online Q-network.
            target = QNetwork(obs_dim, action_dim).to(device)  # Target Q-network.
            target.load_state_dict(net.state_dict())  # Sync target.
            q_nets.append(net)  # Track Q-net.
            target_nets.append(target)  # Track target net.
            optimizers.append(optim.Adam(net.parameters(), lr=dqn_cfg.lr))  # Optimizer.
            buffers.append(ReplayBuffer(dqn_cfg.buffer_size, obs_dim))  # Replay buffer.

    # 3) Training loop state.
    global_step = 0  # Total steps across training.
    episode = 0  # Episode counter.
    episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)  # Accumulated rewards.

    # 4) Main training loop.
    while global_step < args.total_steps:
        # Epsilon-greedy exploration schedule.
        epsilon = linear_schedule(dqn_cfg.epsilon_start, dqn_cfg.epsilon_final, global_step, dqn_cfg.epsilon_decay_steps)  # Exploration rate.
        actions = np.zeros(cfg.n_agents, dtype=np.int64)  # Action array.

        # Select actions for each agent (random with prob epsilon, else greedy).
        for i in range(cfg.n_agents):
            if random.random() < epsilon:
                actions[i] = np.random.randint(0, action_dim)  # Explore.
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)  # Batch-1 obs.
                    q_vals = q_nets[i](obs_tensor)  # Q-values.
                    actions[i] = int(torch.argmax(q_vals, dim=1).item())  # Greedy action.

        # Agent sees observation, picks action, gets reward, environment changes.
        # Step the environment once and record transition data.
        action_dict = _array_to_dict(actions, agent_ids)  # Array -> dict (PettingZoo).
        next_obs_dict, rewards_dict, terminations, truncations, info = env.step(action_dict)  # Env step.
        next_obs = _dict_to_array(next_obs_dict, agent_ids, dtype=np.float32)  # Dict -> array.
        rewards = _dict_to_array(rewards_dict, agent_ids, dtype=np.float32)  # Dict -> array.
        terminated = any(terminations.values())  # Episode ended (success).
        truncated = any(truncations.values())  # Episode ended (time limit).
        done_flag = float(terminated or truncated)  # Done flag for training target.

        # Store transitions in each agent's replay buffer.
        for i in range(cfg.n_agents):
            buffers[i].add(obs[i], actions[i], rewards[i], next_obs[i], done_flag)
            episode_rewards[i] += rewards[i]

        # Move to next observation for the next step.
        obs = next_obs
        global_step += 1

        # 5) Start learning after warmup (collecting initial experience).
        if global_step > dqn_cfg.warmup_steps:
            for i in range(cfg.n_agents):
                if buffers[i].size < dqn_cfg.batch_size:
                    continue
                # Off-policy learning: sample random past transitions.
                batch = buffers[i].sample(dqn_cfg.batch_size)
                batch = [b.to(device) for b in batch]
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = batch

                # Current Q-values for chosen actions.
                q_vals = q_nets[i](b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)  # Q(s,a).
                with torch.no_grad():
                    # Bellman target uses max Q from target network.
                    max_next = target_nets[i](b_next_obs).max(dim=1)[0]  # max_a' Q_target(s',a')
                    target = b_rewards + dqn_cfg.gamma * (1.0 - b_dones) * max_next  # Bellman target.

                # Smooth L1 (Huber) loss stabilizes training vs pure MSE.
                loss = nn.functional.smooth_l1_loss(q_vals, target)  # Huber loss.
                optimizers[i].zero_grad()  # Clear gradients.
                loss.backward()  # Backprop.
                optimizers[i].step()  # Update weights.

        # 6) Periodically sync target networks.
        if global_step % dqn_cfg.target_update == 0:
            for i in range(cfg.n_agents):
                target_nets[i].load_state_dict(q_nets[i].state_dict())  # Sync target net.

        # 7) Episode bookkeeping + logging.
        if terminated or truncated:  # Episode ended.
            episode += 1
            print(
                f"episode {episode} step {global_step} rewards {episode_rewards.mean():.2f} "
                f"targets {info.get('targets_collected', 0)} epsilon {epsilon:.2f}"
            )
            obs_dict, _ = env.reset(seed=args.seed)  # Reset environment.
            obs = _dict_to_array(obs_dict, agent_ids, dtype=np.float32)  # Dict -> array.
            episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)  # Reset reward tracker.

        # 8) Optional checkpointing.
        if args.save_every > 0 and global_step % args.save_every == 0:  # Periodic checkpoint.
            _save_models(args.save_dir, q_nets, args.shared_policy)

    # Final save at end of training.
    _save_models(args.save_dir, q_nets, args.shared_policy)  # Final save.
    env.close()  # Cleanup env.


def parse_args(argv=None):
    """Parse CLI args for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=10000)  # Training steps.
    parser.add_argument("--n-agents", type=int, default=6)  # Number of agents.
    parser.add_argument("--shared-policy", action="store_true")  # Shared vs independent.
    parser.add_argument("--headless", action="store_true")  # No render window.
    parser.add_argument("--cuda", action="store_true")  # Use GPU if available.
    parser.add_argument("--seed", type=int, default=0)  # RNG seed.
    parser.add_argument("--save-dir", type=str, default="checkpoints")  # Checkpoint dir.
    parser.add_argument("--save-every", type=int, default=0, help="Save every N steps (0 = only at end)")  # Save cadence.
    return parser.parse_args(argv)


def _save_models(save_dir: str, q_nets: List[QNetwork], shared: bool):
    """Save model weights to disk, shared or per-agent."""
    os.makedirs(save_dir, exist_ok=True)  # Ensure dir exists.
    if shared:
        torch.save(q_nets[0].state_dict(), os.path.join(save_dir, "shared.pt"))  # One shared file.
        return
    for i, net in enumerate(q_nets):
        torch.save(net.state_dict(), os.path.join(save_dir, f"agent_{i}.pt"))  # Per-agent file.


if __name__ == "__main__":
    train(parse_args())  # Entry point.
