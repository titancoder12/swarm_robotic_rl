from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwarmConfig:
    # World
    width: int = 900
    height: int = 600
    n_agents: int = 6
    n_targets: int = 4
    n_obstacles: int = 6
    agent_radius: float = 10.0
    target_radius: float = 10.0
    max_steps: int = 600

    # Action space (discrete)
    action_dim: int = 1
    num_actions: int = 9

    # Dynamics
    dt: float = 0.1
    max_speed: float = 120.0
    max_yaw_rate: float = 2.5
    accel: float = 300.0
    ang_accel: float = 8.0
    dynamics_mode: str = "tank"  # "tank", "hover", or "mixed"

    # Hovercraft dynamics
    hover_lat_damping: float = 0.85
    hover_lat_noise: float = 5.0
    hover_slip_chance: float = 0.08
    hover_slip_scale: float = 0.6

    # Sensors
    lidar_rays: int = 9
    lidar_max_range: float = 160.0
    lidar_step: float = 6.0
    obs_include_pheromone: bool = True
    pheromone_samples: int = 3

    # Pheromone grid
    pheromone_enabled: bool = True
    pheromone_cell_size: int = 6
    pheromone_deposit: float = 1.0
    pheromone_decay: float = 0.985
    pheromone_diffuse_rate: float = 0.25

    # Rewards
    reward_target: float = 8.0
    reward_step: float = -0.01
    reward_collision: float = -0.2

    # Rendering
    render_pheromone: bool = True
    render_scale: float = 1.0

    # Seeding
    seed: int | None = None
