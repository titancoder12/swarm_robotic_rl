from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv


def main():
    cfg = SwarmConfig()
    env = SwarmEnv(cfg, headless=False)
    obs, info = env.reset(seed=cfg.seed)

    print("obs shape:", obs.shape, "info:", info)
    assert obs.shape == (cfg.n_agents, obs.shape[1])

    total_rewards = np.zeros(cfg.n_agents, dtype=np.float32)
    for step in range(300):
        actions = np.random.randint(0, cfg.num_actions, size=(cfg.n_agents,))
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        if np.isnan(rewards).any():
            raise ValueError("NaN rewards detected")
        if next_obs.shape != obs.shape:
            raise ValueError(f"Observation shape mismatch: {next_obs.shape} vs {obs.shape}")

        total_rewards += rewards
        obs = next_obs

        if step % 30 == 0:
            print(f"step {step} rewards {rewards} info {info}")

        env.render(fps=60)
        if terminated or truncated:
            print("Episode ended", {"terminated": terminated, "truncated": truncated})
            break

    print("total rewards", total_rewards)
    env.close()


if __name__ == "__main__":
    main()
