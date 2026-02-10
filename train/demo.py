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


def parse_args(argv=None):
    # 1) Parse CLI args (checkpoint location, backend, shared policy flag, agent count, seed).
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["custom", "sb3", "rllib"], default="custom")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--sb3-model", type=str, default="checkpoints/sb3_dqn.zip")
    parser.add_argument("--rllib-checkpoint", type=str, default="checkpoints/rllib_dqn")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


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


def _custom_demo(env, obs, agent_ids, args):
    # Infer model dimensions and device.
    obs_dim = obs.shape[1]
    action_dim = env.cfg.num_actions
    device = torch.device("cpu")

    nets = load_models(args.checkpoint_dir, obs_dim, action_dim, env.cfg.n_agents, args.shared_policy, device)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = np.zeros(env.cfg.n_agents, dtype=np.int64)
        for i in range(env.cfg.n_agents):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = nets[i](obs_tensor)
                actions[i] = int(torch.argmax(q_vals, dim=1).item())

        action_dict = {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
        obs_dict, _, terminations, truncations, _ = env.step(action_dict)
        obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        env.render(fps=60)
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=args.seed)
            obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)

    return obs


def _sb3_demo(env, obs_dict, agent_ids, args):
    from stable_baselines3 import DQN

    model = DQN.load(args.sb3_model, device="cpu")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action_dict = {}
        for agent in agent_ids:
            action, _ = model.predict(obs_dict[agent], deterministic=True)
            action_dict[agent] = int(action)

        obs_dict, _, terminations, truncations, _ = env.step(action_dict)
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        env.render(fps=60)
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=args.seed)

    return obs_dict


def _rllib_demo(env, obs_dict, agent_ids, args):
    import os

    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    import ray.rllib.algorithms.algorithm as alg_module
    import ray.rllib.algorithms.dqn.dqn as dqn_module
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    try:
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False, _skip_env_hook=True)
    except TypeError:
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

    # Patch RLlib old-stack replay buffer type handling (coerce class -> string).
    orig_create = alg_module.Algorithm._create_local_replay_buffer_if_necessary

    def _patched_create(self, cfg):
        rb_cfg = cfg.get("replay_buffer_config", {})
        rb_type = rb_cfg.get("type")
        if isinstance(rb_type, type):
            rb_cfg["type"] = rb_type.__name__
        return orig_create(self, cfg)

    alg_module.Algorithm._create_local_replay_buffer_if_necessary = _patched_create

    # Bypass strict validation for replay buffer type on this Ray version.
    dqn_module.DQNConfig.validate = lambda self: None

    def env_creator(_):
        cfg = SwarmConfig(n_agents=args.n_agents)
        return ParallelPettingZooEnv(SwarmEnv(cfg, headless=False))

    register_env("swarm_pz", env_creator)
    checkpoint_path = args.rllib_checkpoint
    if not checkpoint_path.startswith(("file://", "s3://", "gs://")):
        checkpoint_path = os.path.abspath(checkpoint_path)
        checkpoint_path = f"file://{checkpoint_path}"
    algo = Algorithm.from_checkpoint(checkpoint_path)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action_dict = {}
        for agent in agent_ids:
            action = algo.compute_single_action(obs_dict[agent], policy_id="shared_policy")
            action_dict[agent] = int(action)

        obs_dict, _, terminations, truncations, _ = env.step(action_dict)
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        env.render(fps=60)
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=args.seed)

    algo.stop()
    ray.shutdown()
    return obs_dict


def main():
    args = parse_args()

    # 2) Build config + environment, then reset to get initial observations.
    cfg = SwarmConfig(n_agents=args.n_agents)
    env = SwarmEnv(cfg, headless=False)
    obs_dict, _ = env.reset(seed=args.seed)
    agent_ids = env.possible_agents
    obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
    env.render(fps=60) # render first frame

    if args.backend == "custom":
        _custom_demo(env, obs, agent_ids, args)
    elif args.backend == "sb3":
        _sb3_demo(env, obs_dict, agent_ids, args)
    elif args.backend == "rllib":
        _rllib_demo(env, obs_dict, agent_ids, args)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    env.close()


if __name__ == "__main__":
    main()
