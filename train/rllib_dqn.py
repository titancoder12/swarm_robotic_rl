from __future__ import annotations

import argparse
import os
import sys

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

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
    parser.add_argument("--save-dir", type=str, default="checkpoints/rllib_dqn")
    return parser.parse_args(argv)


def run(args):
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    def env_creator(_):
        cfg = SwarmConfig(n_agents=args.n_agents)
        return ParallelPettingZooEnv(SwarmEnv(cfg, headless=args.headless))

    register_env("swarm_pz", env_creator)

    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space("agent_0")
    act_space = tmp_env.action_space("agent_0")
    tmp_env.close()

    policies = {"shared_policy": (None, obs_space, act_space, {})}
    policy_mapping_fn = lambda agent_id, *_, **__: "shared_policy"

    config = (
        DQNConfig()
        .environment("swarm_pz")
        .framework("torch")
        .rollouts(num_rollout_workers=0)
        .training(seed=args.seed)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    algo = config.build()
    timesteps = 0
    while timesteps < args.total_steps:
        result = algo.train()
        timesteps = result.get("timesteps_total", timesteps)

    os.makedirs(args.save_dir, exist_ok=True)
    algo.save(args.save_dir)
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    run(parse_args())
