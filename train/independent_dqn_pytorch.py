"""
Classic DQN / Q-learning setup (with replay buffer + target network) 
and an ε-greedy behavior policy.

"""
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv


@dataclass
class DQNConfig:
    """Hyperparameters for DQN training."""
    gamma: float = 0.98
    batch_size: int = 64
    buffer_size: int = 50_000
    lr: float = 3e-4
    target_update: int = 200
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 8000
    warmup_steps: int = 500


class ReplayBuffer:
    """Simple FIFO replay buffer for experience tuples."""
    def __init__(self, capacity: int, obs_dim: int):
        """Allocate storage for fixed-size experience arrays."""
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Insert a transition, overwriting oldest when capacity is exceeded."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a random minibatch and return as torch tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.actions[idx]),
            torch.tensor(self.rewards[idx]),
            torch.tensor(self.next_obs[idx]),
            torch.tensor(self.dones[idx]),
        )


class QNetwork(nn.Module):
    """Small MLP mapping observations to Q-values for each discrete action."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        # Two hidden layers with ReLU nonlinearity.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        """Return Q-values for each action."""
        return self.net(x)


def linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    """Linearly anneal a value from start to end over decay_steps."""
    if step >= decay_steps:
        return end
    frac = step / decay_steps
    return start + frac * (end - start)


def train(args):
    """Train independent (or shared) DQN policies for each agent."""
    # 1) Environment and config setup.
    cfg = SwarmConfig(n_agents=args.n_agents)
    dqn_cfg = DQNConfig()

    env = SwarmEnv(cfg, headless=args.headless)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = obs.shape[1]
    action_dim = cfg.num_actions

    # Use GPU if requested and available.
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 2) Initialize Q-networks, target networks, optimizers, and replay buffers.
    if args.shared_policy:
        # One shared policy across all agents.
        q_net = QNetwork(obs_dim, action_dim).to(device)
        target_net = QNetwork(obs_dim, action_dim).to(device)
        target_net.load_state_dict(q_net.state_dict())
        optimizer = optim.Adam(q_net.parameters(), lr=dqn_cfg.lr)
        buffers = [ReplayBuffer(dqn_cfg.buffer_size, obs_dim) for _ in range(cfg.n_agents)]
        q_nets = [q_net for _ in range(cfg.n_agents)]
        target_nets = [target_net for _ in range(cfg.n_agents)]
        optimizers = [optimizer for _ in range(cfg.n_agents)]
    else:
        # Independent policies (one per agent).
        q_nets = []
        target_nets = []
        optimizers = []
        buffers = []
        for _ in range(cfg.n_agents):
            net = QNetwork(obs_dim, action_dim).to(device)
            target = QNetwork(obs_dim, action_dim).to(device)
            target.load_state_dict(net.state_dict())
            q_nets.append(net)
            target_nets.append(target)
            optimizers.append(optim.Adam(net.parameters(), lr=dqn_cfg.lr))
            buffers.append(ReplayBuffer(dqn_cfg.buffer_size, obs_dim))

    # 3) Training loop state.
    global_step = 0
    episode = 0
    episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)

    # 4) Main training loop.
    while global_step < args.total_steps:
        # Epsilon-greedy exploration schedule.
        epsilon = linear_schedule(dqn_cfg.epsilon_start, dqn_cfg.epsilon_final, global_step, dqn_cfg.epsilon_decay_steps)
        actions = np.zeros(cfg.n_agents, dtype=np.int64)

        # Select actions for each agent (random with prob epsilon, else greedy).
        for i in range(cfg.n_agents):
            if random.random() < epsilon:
                actions[i] = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = q_nets[i](obs_tensor)
                    actions[i] = int(torch.argmax(q_vals, dim=1).item())

        # Agent sees observation, picks action, gets reward, environment changes.
        # Step the environment once and record transition data.
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done_flag = float(terminated or truncated)

        # Store transitions in each agent's replay buffer.
        for i in range(cfg.n_agents):
            buffers[i].add(obs[i], actions[i], rewards[i], next_obs[i], done_flag)
            episode_rewards[i] += rewards[i]

        obs = next_obs
        global_step += 1

        # 5) Start learning after warmup (collecting initial experience).
        if global_step > dqn_cfg.warmup_steps:
            for i in range(cfg.n_agents):
                if buffers[i].size < dqn_cfg.batch_size:
                    continue
                batch = buffers[i].sample(dqn_cfg.batch_size)
                batch = [b.to(device) for b in batch]
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = batch

                # Current Q-values for chosen actions.
                q_vals = q_nets[i](b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # Target uses max Q from target network.
                    max_next = target_nets[i](b_next_obs).max(dim=1)[0]
                    target = b_rewards + dqn_cfg.gamma * (1.0 - b_dones) * max_next

                # Smooth L1 (Huber) loss stabilizes training vs pure MSE.
                loss = nn.functional.smooth_l1_loss(q_vals, target)
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

        # 6) Periodically sync target networks.
        if global_step % dqn_cfg.target_update == 0:
            for i in range(cfg.n_agents):
                target_nets[i].load_state_dict(q_nets[i].state_dict())

        # 7) Episode bookkeeping + logging.
        if terminated or truncated:
            episode += 1
            print(
                f"episode {episode} step {global_step} rewards {episode_rewards.mean():.2f} "
                f"targets {info.get('targets_collected', 0)} epsilon {epsilon:.2f}"
            )
            obs, _ = env.reset(seed=args.seed)
            episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)

        # 8) Optional checkpointing.
        if args.save_every > 0 and global_step % args.save_every == 0:
            _save_models(args.save_dir, q_nets, args.shared_policy)

    # Final save at end of training.
    _save_models(args.save_dir, q_nets, args.shared_policy)
    env.close()


def parse_args():
    """Parse CLI args for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=0, help="Save every N steps (0 = only at end)")
    return parser.parse_args()


def _save_models(save_dir: str, q_nets: List[QNetwork], shared: bool):
    """Save model weights to disk, shared or per-agent."""
    os.makedirs(save_dir, exist_ok=True)
    if shared:
        torch.save(q_nets[0].state_dict(), os.path.join(save_dir, "shared.pt"))
        return
    for i, net in enumerate(q_nets):
        torch.save(net.state_dict(), os.path.join(save_dir, f"agent_{i}.pt"))


if __name__ == "__main__":
    train(parse_args())
