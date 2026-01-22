# Project Log

Use this file to capture decisions, open questions, and next steps so we can resume smoothly across sessions.

## Summary
- Multi-agent PyGame swarm environment with stigmergy.
- Discrete 9-action joystick interface.
- DQN training with independent or shared policies.
- Demo and screenshot tooling included.

## Key Commands
- Random rollout: `python train/random_rollout.py`
- Train (headless): `python train/independent_dqn_pytorch.py --headless --total-steps 10000 --save-dir checkpoints --save-every 2000`
- Demo: `python train/demo.py --checkpoint-dir checkpoints`
- Screenshot gallery: `python train/capture_screenshots.py`

## Decisions
- Default training uses independent Q-networks unless `--shared-policy` is passed.
- Pheromone grid uses simple deposit + decay + diffusion.

## Open Questions
- Do we want a nest/return task and food-carrying state?
- Should pheromone deposit depend on carrying food?
- Add video capture or logging dashboards?

## Next Steps
- Add carry state + nest reward (optional)
- Add two pheromone channels (food vs nest)
- Improve training stability or add PPO alternative
