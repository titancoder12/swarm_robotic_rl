from __future__ import annotations

import argparse
import os
import sys

import os

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray
from ray.rllib.algorithms.dqn import DQN, DQNConfig
import ray.rllib.algorithms.dqn.dqn as dqn_module
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import (
    MultiAgentPrioritizedReplayBuffer,
)
from ray.rllib.algorithms.algorithm import Algorithm
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
    parser.add_argument(
        "--ray-tmpdir",
        type=str,
        default="",
        help="Override Ray temp dir (useful to avoid /tmp space or socket path length issues).",
    )
    return parser.parse_args(argv)


def run(args):
    if args.ray_tmpdir:
        os.environ["RAY_TMPDIR"] = args.ray_tmpdir
    try:
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False, _skip_env_hook=True)
    except TypeError:
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

    def env_creator(_):
        cfg = SwarmConfig(n_agents=args.n_agents)
        return ParallelPettingZooEnv(SwarmEnv(cfg, headless=args.headless))

    register_env("swarm_pz", env_creator)

    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space["agent_0"]
    act_space = tmp_env.action_space["agent_0"]
    tmp_env.close()

    policies = {"shared_policy": (None, obs_space, act_space, {})}
    policy_mapping_fn = lambda agent_id, *_, **__: "shared_policy"

    config = (
        DQNConfig()
        .environment("swarm_pz")
        .framework("torch")
        .env_runners(num_env_runners=0)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )
    config = config.experimental(_validate_config=False)
    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    config = config.training(
        replay_buffer_config={
            "type": MultiAgentPrioritizedReplayBuffer,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
        }
    )

    config_dict = config.to_dict()

    # Patch RLlib old-stack replay buffer type handling (coerce class -> string).
    orig_create = Algorithm._create_local_replay_buffer_if_necessary

    def _patched_create(self, cfg):
        rb_cfg = cfg.get("replay_buffer_config", {})
        rb_type = rb_cfg.get("type")
        if isinstance(rb_type, type):
            rb_cfg["type"] = rb_type.__name__
        return orig_create(self, cfg)

    Algorithm._create_local_replay_buffer_if_necessary = _patched_create

    # Bypass strict validation for replay buffer type (Ray version mismatch).
    dqn_module.DQNConfig.validate = lambda self: None
    algo = DQN(config=config_dict)
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
