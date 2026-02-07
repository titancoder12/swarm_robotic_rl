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
    out_dir = os.path.join(ROOT, "docs", "images")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "swarm_demo.png")

    cfg = SwarmConfig()
    env = SwarmEnv(cfg, headless=False)
    obs_dict, _ = env.reset(seed=0)
    agent_ids = env.possible_agents

    # Step a few times to get a non-trivial scene.
    for _ in range(10):
        actions = {agent: int(np.random.randint(0, cfg.num_actions)) for agent in agent_ids}
        obs_dict, _, terminations, truncations, _ = env.step(actions)
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        env.render(fps=60)
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=0)

    env.save_screenshot(out_path)
    env.close()
    print(f"Saved screenshot to {out_path}")


if __name__ == "__main__":
    main()
