from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv
from train.independent_dqn_pytorch import QNetwork


def load_models(checkpoint_dir: str, obs_dim: int, action_dim: int, n_agents: int, shared: bool, device):
    if shared:
        net = QNetwork(obs_dim, action_dim).to(device)
        path = os.path.join(checkpoint_dir, "shared.pt")
        net.load_state_dict(torch.load(path, map_location=device))
        nets = [net for _ in range(n_agents)]
    else:
        nets = []
        for i in range(n_agents):
            net = QNetwork(obs_dim, action_dim).to(device)
            path = os.path.join(checkpoint_dir, f"agent_{i}.pt")
            net.load_state_dict(torch.load(path, map_location=device))
            nets.append(net)
    for net in nets:
        net.eval()
    return nets


def main():
    # 1) Parse CLI args (checkpoint location, shared policy flag, agent count, seed).
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 2) Build config + environment, then reset to get initial observations.
    cfg = SwarmConfig(n_agents=args.n_agents)
    env = SwarmEnv(cfg, headless=False)
    obs, _ = env.reset(seed=args.seed)
    env.render(fps=60) # render first frame

    # 3) Infer model dimensions and device.
    obs_dim = obs.shape[1]
    action_dim = cfg.num_actions
    device = torch.device("cpu")

    # 4) Load policy network(s) from checkpoints.
    nets = load_models(args.checkpoint_dir, obs_dim, action_dim, cfg.n_agents, args.shared_policy, device)

    # 5) Main loop: handle events, infer actions, step env, render, reset if done.
    running = True
    while running:
        # Handle PyGame window events (including close).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Policy inference: choose discrete action for each agent.
        actions = np.zeros(cfg.n_agents, dtype=np.int64)
        for i in range(cfg.n_agents):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = nets[i](obs_tensor)
                actions[i] = int(torch.argmax(q_vals, dim=1).item())

        # Step the environment (updates state + returns new observations).
        obs, _, terminated, truncated, _ = env.step(actions)
        # Render the updated state.
        env.render(fps=60)
        # Reset episode if finished.
        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed)

    # 6) Clean up PyGame resources on exit.
    env.close()


if __name__ == "__main__":
    main()
