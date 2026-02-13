# Sim-to-Real To-Do (Required Changes)

This list captures the minimum changes needed to move from simulation to a physical robot deployment.

## 1) Observation Pipeline (Sensors → `obs`)
- Replace `_get_obs()` with a real sensor fusion pipeline that outputs the **same 19‑dim vector** (or retrain with a new vector).
- Map lidar rays to actual range sensors or 2D LiDAR.
- Replace nearest target/neighbor vectors with real detection or estimation.
- Compute heading and speed from IMU/odometry.
- Replace pheromone samples with a real signal (or disable that channel).

## 2) Action Interface (Policy → Motors)
- Map discrete actions (9 joystick commands) to real motor commands.
- Implement safety limits (speed, acceleration, turn rate).
- Add a low‑level controller for smooth actuation.

## 3) Target Detection (Reward / Success)
- Define how a robot detects “target collected” (vision marker, RFID, IR beacon, contact switch).
- Replace the sim oracle in `_handle_targets()` with sensor‑based detection.
- Log events for reward and evaluation.

## 4) Episode / Reset Mechanics
- In sim, `reset()` respawns targets/agents instantly.
- In real world, define a **manual or automated reset** procedure.
- Add a time limit and safe stop behavior for truncation.

## 5) Safety & Fault Handling
- Obstacle avoidance failsafe (emergency stop on collision risk).
- Battery and temperature checks.
- Network disconnect and sensor failure handling.

## 6) Domain Gap Mitigation
- Add noise/randomization in sim (sensor noise, dynamics variations).
- Calibrate dynamics and observation scales to match the real robot.
- Use randomized target positions, lighting, and textures if vision is involved.
- Consider curriculum training (simple → complex) to improve robustness.

## 7) Deployment Runtime
- Create a real-time loop that runs policy inference at a fixed frequency.
- Ensure deterministic inference timing and logging.

## 8) Data Logging & Evaluation
- Log sensor streams, actions, and events for debugging.
- Define success metrics and test protocols.

## 9) Sim-to-Real Training Checks
- Verify observations are **normalized the same way** in sim and real.
- Ensure action scaling/limits match the real robot.
- Evaluate the policy in sim with **noise and delays** similar to real sensors.
- Add safety constraints during training (e.g., penalty for unsafe proximity).
