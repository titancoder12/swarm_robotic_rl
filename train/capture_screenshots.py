from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv


def _run_scene(cfg: SwarmConfig, out_path: str, steps: int = 15):
    env = SwarmEnv(cfg, headless=False)
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        actions = np.random.randint(0, cfg.num_actions, size=(cfg.n_agents,))
        obs, _, terminated, truncated, _ = env.step(actions)
        env.render(fps=60)
        if terminated or truncated:
            obs, _ = env.reset(seed=0)
    env.save_screenshot(out_path)
    env.close()


def main():
    out_dir = os.path.join(ROOT, "docs", "images")
    os.makedirs(out_dir, exist_ok=True)

    scenes: List[Tuple[str, SwarmConfig]] = []

    cfg_default = SwarmConfig()
    scenes.append(("swarm_default.png", cfg_default))

    cfg_no_pher = SwarmConfig(render_pheromone=False)
    scenes.append(("swarm_no_pheromone.png", cfg_no_pher))

    cfg_hover = SwarmConfig(dynamics_mode="hover")
    scenes.append(("swarm_hover.png", cfg_hover))

    cfg_dense = SwarmConfig(n_agents=10, n_targets=6, n_obstacles=8)
    scenes.append(("swarm_dense.png", cfg_dense))

    for filename, cfg in scenes:
        out_path = os.path.join(out_dir, filename)
        _run_scene(cfg, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
