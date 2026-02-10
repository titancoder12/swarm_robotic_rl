from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import supersuit as ss
import torch
from stable_baselines3 import DQN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-path", type=str, default="checkpoints/sb3_dqn.zip")
    return parser.parse_args(argv)


def run(args):
    cfg = SwarmConfig(n_agents=args.n_agents)
    base_env = SwarmEnv(cfg, headless=args.headless)

    env = ss.pettingzoo_env_to_vec_env_v1(base_env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        env.reset()
    except TypeError:
        pass

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=None)
    model.learn(total_timesteps=args.total_steps)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)
    env.close()


if __name__ == "__main__":
    run(parse_args())
