from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

from env.config import SwarmConfig


@dataclass
class AgentState:
    """Lightweight container for an agent's kinematic state in the world."""
    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0
    v_lat: float = 0.0


class DynamicsDriver:
    """Interface for mapping actions into kinematic updates."""
    def apply(self, state: AgentState, action: Tuple[float, float], dt: float, cfg: SwarmConfig, rng: np.random.Generator) -> AgentState:
        """Return the next AgentState given the current state and action."""
        raise NotImplementedError


class TankKinematicsDriver(DynamicsDriver):
    """Differential-drive style dynamics (forward speed + yaw rate)."""
    def apply(self, state: AgentState, action: Tuple[float, float], dt: float, cfg: SwarmConfig, rng: np.random.Generator) -> AgentState:
        """Apply tank kinematics with acceleration limits."""
        throttle, turn = action
        target_v = throttle * cfg.max_speed
        target_omega = turn * cfg.max_yaw_rate
        dv = np.clip(target_v - state.v, -cfg.accel * dt, cfg.accel * dt)
        domega = np.clip(target_omega - state.omega, -cfg.ang_accel * dt, cfg.ang_accel * dt)

        v = state.v + dv
        omega = state.omega + domega
        theta = state.theta + omega * dt

        nx = state.x + math.cos(theta) * v * dt
        ny = state.y + math.sin(theta) * v * dt
        return AgentState(nx, ny, theta, v=v, omega=omega, v_lat=0.0)


class HovercraftDriver(DynamicsDriver):
    """Hover-like dynamics with lateral drift and slip."""
    def apply(self, state: AgentState, action: Tuple[float, float], dt: float, cfg: SwarmConfig, rng: np.random.Generator) -> AgentState:
        """Apply hovercraft kinematics including lateral noise and slip."""
        throttle, turn = action
        target_v = throttle * cfg.max_speed
        target_omega = turn * cfg.max_yaw_rate
        dv = np.clip(target_v - state.v, -cfg.accel * dt, cfg.accel * dt)
        domega = np.clip(target_omega - state.omega, -cfg.ang_accel * dt, cfg.ang_accel * dt)

        v = state.v + dv
        omega = state.omega + domega
        theta = state.theta + omega * dt

        v_lat = state.v_lat * cfg.hover_lat_damping
        v_lat += rng.normal(0.0, cfg.hover_lat_noise)
        if rng.random() < cfg.hover_slip_chance:
            v *= cfg.hover_slip_scale

        forward = np.array([math.cos(theta), math.sin(theta)])
        right = np.array([math.cos(theta + math.pi / 2.0), math.sin(theta + math.pi / 2.0)])
        vel = forward * v + right * v_lat

        nx = state.x + vel[0] * dt
        ny = state.y + vel[1] * dt
        return AgentState(nx, ny, theta, v=v, omega=omega, v_lat=v_lat)


class SwarmEnv:
    """Multi-agent 2D swarm environment with optional stigmergy and PyGame rendering."""
    def __init__(self, cfg: SwarmConfig, headless: bool = False):
        """Create an environment instance with a given config."""
        self.cfg = cfg
        self.headless = headless
        self.rng = np.random.default_rng(cfg.seed)

        self.width = cfg.width
        self.height = cfg.height

        self.action_table = self._build_action_table()
        self.driver_mode = cfg.dynamics_mode
        self.driver = self._select_driver(cfg.dynamics_mode)

        self.agents: List[AgentState] = []
        self.targets: List[Tuple[float, float]] = []
        self.obstacles: List[pygame.Rect] = []

        self.step_count = 0
        self.terminated = False
        self.truncated = False

        self.pheromone_grid = None

        self._pygame_inited = False
        self._screen = None
        self._clock = None

    def reset(self, seed: int | None = None):
        """Reset the environment and return initial observations and info."""
        # (Re)initialize RNG and episode state, then spawn a fresh world.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        # Randomize dynamics driver if mixed mode is enabled.
        if self.cfg.dynamics_mode == "mixed":
            self.driver = self._select_driver(self.rng.choice(["tank", "hover"]))
        else:
            self.driver = self._select_driver(self.cfg.dynamics_mode)

        # World state: obstacles, targets, agents.
        self._spawn_obstacles()
        self._spawn_targets()
        self._spawn_agents()

        # Optional pheromone grid for stigmergy.
        if self.cfg.pheromone_enabled:
            grid_w = self.width // self.cfg.pheromone_cell_size + 1
            grid_h = self.height // self.cfg.pheromone_cell_size + 1
            self.pheromone_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        else:
            self.pheromone_grid = None

        obs = self._get_obs()
        info = {"n_targets": len(self.targets)}
        return obs, info

    def step(self, actions: np.ndarray):
        """Advance the simulation by one step using agent actions."""
        # One environment tick: apply actions, move agents, compute rewards/obs.
        actions = np.asarray(actions)
        if actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
        if actions.shape != (self.cfg.n_agents,):
            raise ValueError(f"actions must have shape (n_agents,) or (n_agents,1). Got {actions.shape}")

        # Start with per-step reward for all agents.
        rewards = np.full((self.cfg.n_agents,), self.cfg.reward_step, dtype=np.float32)
        collisions = 0

        for i, agent in enumerate(self.agents):
            action_id = int(actions[i])
            throttle, turn = self.action_table[action_id]
            # Propose next state from dynamics, then check collisions.
            proposed = self.driver.apply(agent, (throttle, turn), self.cfg.dt, self.cfg, self.rng)

            collided = self._handle_collisions(proposed)
            if collided:
                rewards[i] += self.cfg.reward_collision
                collisions += 1
            else:
                self.agents[i] = proposed

        # Handle target collection and pheromone updates.
        collected = self._handle_targets(rewards)

        if self.cfg.pheromone_enabled:
            self._update_pheromone()

        # Episode end conditions.
        self.step_count += 1
        if len(self.targets) == 0:
            self.terminated = True
        if self.step_count >= self.cfg.max_steps:
            self.truncated = True

        obs = self._get_obs()
        info = {"targets_collected": collected, "collisions": collisions}
        return obs, rewards, self.terminated, self.truncated, info

    def render(self, mode: str = "human", fps: int = 60):
        """Render the current state to a PyGame window."""
        # Draw the current world state to a PyGame window.
        if self.headless:
            return
        self._ensure_pygame()

        # Background.
        self._screen.fill((20, 20, 26))

        if self.cfg.pheromone_enabled and self.cfg.render_pheromone:
            self._draw_pheromone()

        # Obstacles.
        for rect in self.obstacles:
            pygame.draw.rect(self._screen, (70, 70, 80), rect)

        # Targets.
        for tx, ty in self.targets:
            pygame.draw.circle(self._screen, (80, 200, 80), (int(tx), int(ty)), int(self.cfg.target_radius))

        # Agents (body + heading line).
        for agent in self.agents:
            x, y = int(agent.x), int(agent.y)
            pygame.draw.circle(self._screen, (200, 160, 50), (x, y), int(self.cfg.agent_radius))
            hx = x + int(math.cos(agent.theta) * self.cfg.agent_radius)
            hy = y + int(math.sin(agent.theta) * self.cfg.agent_radius)
            pygame.draw.line(self._screen, (255, 240, 180), (x, y), (hx, hy), 2)

        # Present frame and limit FPS.
        pygame.display.flip()
        self._clock.tick(fps)

    def save_screenshot(self, path: str):
        """Save the current render buffer to an image file."""
        if self.headless:
            raise RuntimeError("Cannot save screenshot in headless mode.")
        self._ensure_pygame()
        pygame.image.save(self._screen, path)

    def close(self):
        """Shut down PyGame resources if they were initialized."""
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False

    def _ensure_pygame(self):
        """Initialize PyGame on first render, no-op if already initialized."""
        # Lazy-init PyGame so headless training doesn't open a window.
        if not self._pygame_inited:
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Swarm RL")
            self._clock = pygame.time.Clock()
            self._pygame_inited = True

    def _build_action_table(self):
        """Create the discrete action lookup table (throttle, turn)."""
        throttle_vals = [-1.0, 0.0, 1.0]
        turn_vals = [-1.0, 0.0, 1.0]
        table = []
        for throttle in throttle_vals:
            for turn in turn_vals:
                table.append((throttle, turn))
        return table

    def _select_driver(self, mode: str) -> DynamicsDriver:
        """Return a dynamics driver based on the configured mode."""
        if mode == "hover":
            return HovercraftDriver()
        return TankKinematicsDriver()

    def _spawn_agents(self):
        """Randomly place agents in non-colliding free space."""
        # Randomly place agents in free space.
        self.agents = []
        for _ in range(self.cfg.n_agents):
            pos = self._sample_free_position(self.cfg.agent_radius)
            theta = self.rng.uniform(-math.pi, math.pi)
            self.agents.append(AgentState(pos[0], pos[1], theta))

    def _spawn_targets(self):
        """Randomly place targets in non-colliding free space."""
        # Randomly place targets in free space.
        self.targets = []
        for _ in range(self.cfg.n_targets):
            pos = self._sample_free_position(self.cfg.target_radius)
            self.targets.append(pos)

    def _spawn_obstacles(self):
        """Create random obstacle rectangles with simple overlap avoidance."""
        # Randomly generate rectangular obstacles without overlaps.
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.cfg.n_obstacles and attempts < 200:
            attempts += 1
            w = self.rng.integers(60, 120)
            h = self.rng.integers(40, 120)
            x = self.rng.integers(30, self.width - 30 - w)
            y = self.rng.integers(30, self.height - 30 - h)
            rect = pygame.Rect(int(x), int(y), int(w), int(h))
            if any(rect.colliderect(o) for o in self.obstacles):
                continue
            self.obstacles.append(rect)

    def _sample_free_position(self, radius: float):
        """Sample a random position that does not overlap obstacles or entities."""
        for _ in range(200):
            x = self.rng.uniform(radius, self.width - radius)
            y = self.rng.uniform(radius, self.height - radius)
            circle = pygame.Rect(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
            if any(circle.colliderect(o) for o in self.obstacles):
                continue
            if any((ax - x) ** 2 + (ay - y) ** 2 < (radius * 2) ** 2 for ax, ay in self.targets):
                continue
            if any((agent.x - x) ** 2 + (agent.y - y) ** 2 < (radius * 2) ** 2 for agent in self.agents):
                continue
            return x, y
        return radius, radius

    def _handle_collisions(self, proposed: AgentState) -> bool:
        """Return True if the proposed state collides with bounds/obstacles."""
        # Check arena bounds and obstacle collision for a proposed state.
        if proposed.x < self.cfg.agent_radius or proposed.x > self.width - self.cfg.agent_radius:
            return True
        if proposed.y < self.cfg.agent_radius or proposed.y > self.height - self.cfg.agent_radius:
            return True
        agent_rect = pygame.Rect(
            int(proposed.x - self.cfg.agent_radius),
            int(proposed.y - self.cfg.agent_radius),
            int(self.cfg.agent_radius * 2),
            int(self.cfg.agent_radius * 2),
        )
        return any(agent_rect.colliderect(o) for o in self.obstacles)

    def _handle_targets(self, rewards: np.ndarray) -> int:
        """Assign rewards for collected targets and remove them from the world."""
        # Assign reward when an agent reaches a target, then remove it.
        collected = 0
        remaining = []
        for tx, ty in self.targets:
            collected_by = None
            for i, agent in enumerate(self.agents):
                if (agent.x - tx) ** 2 + (agent.y - ty) ** 2 <= (self.cfg.target_radius + self.cfg.agent_radius) ** 2:
                    collected_by = i
                    break
            if collected_by is not None:
                rewards[collected_by] += self.cfg.reward_target
                collected += 1
            else:
                remaining.append((tx, ty))
        self.targets = remaining
        return collected

    def _update_pheromone(self):
        """Deposit, decay, and diffuse pheromone values."""
        # Deposit pheromone where agents are, then decay/diffuse the grid.
        grid = self.pheromone_grid
        cell = self.cfg.pheromone_cell_size
        for agent in self.agents:
            gx = int(agent.x // cell)
            gy = int(agent.y // cell)
            if 0 <= gy < grid.shape[0] and 0 <= gx < grid.shape[1]:
                grid[gy, gx] += self.cfg.pheromone_deposit

        grid *= self.cfg.pheromone_decay
        diff = self.cfg.pheromone_diffuse_rate
        if diff > 0:
            up = np.roll(grid, 1, axis=0)
            down = np.roll(grid, -1, axis=0)
            left = np.roll(grid, 1, axis=1)
            right = np.roll(grid, -1, axis=1)
            neighbor_avg = (up + down + left + right) * 0.25
            grid[:] = grid * (1.0 - diff) + neighbor_avg * diff

    def _get_obs(self):
        """Assemble per-agent observations (lidar, targets, neighbors, etc.)."""
        # Build per-agent observation vectors.
        obs_list = []
        for idx, agent in enumerate(self.agents):
            lidar = self._lidar_scan(agent)
            target_vec = self._nearest_target_vector(agent)
            neighbor_vec = self._nearest_agent_vector(agent, idx)
            heading = np.array([math.sin(agent.theta), math.cos(agent.theta)], dtype=np.float32)
            speed = np.array([np.clip(agent.v / self.cfg.max_speed, -1.0, 1.0)], dtype=np.float32)
            pheromone = self._pheromone_samples(agent)

            obs = np.concatenate([lidar, target_vec, neighbor_vec, heading, speed, pheromone]).astype(np.float32)
            obs_list.append(obs)
        return np.stack(obs_list, axis=0)

    def _lidar_scan(self, agent: AgentState) -> np.ndarray:
        """Return normalized lidar ray distances for a single agent."""
        # Cast lidar rays and return normalized distances.
        rays = []
        half = self.cfg.lidar_rays // 2
        for i in range(self.cfg.lidar_rays):
            angle = agent.theta + (i - half) * (math.pi / (self.cfg.lidar_rays - 1))
            dist = self._ray_distance(agent.x, agent.y, angle)
            rays.append(dist / self.cfg.lidar_max_range)
        return np.array(rays, dtype=np.float32)

    def _ray_distance(self, x: float, y: float, angle: float) -> float:
        """Return distance from (x, y) to nearest obstacle/boundary along a ray."""
        # March a ray forward until it hits an obstacle or max range.
        max_range = self.cfg.lidar_max_range
        step = self.cfg.lidar_step
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dist = 0.0
        while dist < max_range:
            px = x + cos_a * dist
            py = y + sin_a * dist
            if px < 0 or px > self.width or py < 0 or py > self.height:
                return dist
            point_rect = pygame.Rect(int(px), int(py), 2, 2)
            if any(point_rect.colliderect(o) for o in self.obstacles):
                return dist
            dist += step
        return max_range

    def _nearest_target_vector(self, agent: AgentState) -> np.ndarray:
        """Return nearest target vector in agent-local coordinates."""
        # Vector from agent to nearest target, in agent-local coordinates.
        if not self.targets:
            return np.zeros(2, dtype=np.float32)
        targets = np.array(self.targets)
        dx = targets[:, 0] - agent.x
        dy = targets[:, 1] - agent.y
        dists = np.hypot(dx, dy)
        idx = int(np.argmin(dists))
        rel = np.array([dx[idx], dy[idx]], dtype=np.float32)
        rel = self._to_agent_frame(rel, agent.theta)
        return np.clip(rel / self.cfg.lidar_max_range, -1.0, 1.0)

    def _nearest_agent_vector(self, agent: AgentState, idx: int) -> np.ndarray:
        """Return nearest neighbor vector in agent-local coordinates."""
        # Vector from agent to nearest neighbor, in agent-local coordinates.
        if self.cfg.n_agents <= 1:
            return np.zeros(2, dtype=np.float32)
        best = None
        best_dist = float("inf")
        for j, other in enumerate(self.agents):
            if j == idx:
                continue
            dx = other.x - agent.x
            dy = other.y - agent.y
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best = np.array([dx, dy], dtype=np.float32)
        if best is None:
            return np.zeros(2, dtype=np.float32)
        rel = self._to_agent_frame(best, agent.theta)
        return np.clip(rel / self.cfg.lidar_max_range, -1.0, 1.0)

    def _pheromone_samples(self, agent: AgentState) -> np.ndarray:
        """Sample pheromone values along the agent's forward direction."""
        # Sample pheromone intensity in front of the agent.
        if not (self.cfg.pheromone_enabled and self.cfg.obs_include_pheromone):
            return np.zeros(self.cfg.pheromone_samples, dtype=np.float32)
        grid = self.pheromone_grid
        cell = self.cfg.pheromone_cell_size
        samples = []
        for i in range(self.cfg.pheromone_samples):
            dist = (i + 1) * self.cfg.agent_radius * 1.5
            sx = agent.x + math.cos(agent.theta) * dist
            sy = agent.y + math.sin(agent.theta) * dist
            gx = int(np.clip(sx // cell, 0, grid.shape[1] - 1))
            gy = int(np.clip(sy // cell, 0, grid.shape[0] - 1))
            samples.append(grid[gy, gx])
        samples = np.array(samples, dtype=np.float32)
        if samples.max() > 0:
            samples = samples / (samples.max() + 1e-6)
        return samples

    def _to_agent_frame(self, vec: np.ndarray, theta: float) -> np.ndarray:
        """Rotate a world-space vector into the agent's local frame."""
        # Rotate a world-space vector into the agent's local frame.
        c = math.cos(-theta)
        s = math.sin(-theta)
        x, y = vec
        return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)

    def _draw_pheromone(self):
        """Render a heatmap-style visualization of the pheromone grid."""
        # Render pheromone heatmap as colored grid cells.
        grid = self.pheromone_grid
        if grid is None:
            return
        cell = self.cfg.pheromone_cell_size
        max_val = grid.max()
        if max_val <= 0:
            return
        for gy in range(grid.shape[0]):
            for gx in range(grid.shape[1]):
                val = grid[gy, gx] / max_val
                if val <= 0.01:
                    continue
                color = (int(40 + 160 * val), int(40 + 40 * val), int(80 + 120 * val))
                rect = pygame.Rect(gx * cell, gy * cell, cell, cell)
                self._screen.fill(color, rect)


if __name__ == "__main__":
    cfg = SwarmConfig()
    env = SwarmEnv(cfg)
    obs, info = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        actions = np.random.randint(0, cfg.num_actions, size=(cfg.n_agents,))
        env.step(actions)
        env.render()
    env.close()
