# Repository Guidelines

## Project Structure & Module Organization
- `env/`: core environment implementation (`env/swarm_env.py`) and configuration (`env/config.py`).
- `train/`: runnable scripts for rollouts, training, demos, and screenshots.
- `docs/`: architecture and project log docs plus `docs/images/` assets.
- `checkpoints/`: default location for saved model checkpoints.

## Build, Test, and Development Commands
- `python -m venv .venv` and `source .venv/bin/activate`: create/activate a local virtual env.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python train/random_rollout.py`: rendered random policy sanity check.
- `python train/independent_dqn_pytorch.py --headless --total-steps 10000`: headless training run.
- `python train/independent_dqn_pytorch.py --headless --save-every 2000 --save-dir checkpoints`: save checkpoints during training.
- `python train/train.py --backend sb3 --headless --total-steps 10000`: SB3 DQN training (shared policy).
- `python train/train.py --backend rllib --headless --total-steps 10000`: RLlib DQN training (shared policy).
- If Ray warns about `/tmp` space, pass `--ray-tmpdir` to the RLlib commands.
- `python train/demo.py --checkpoint-dir checkpoints`: render a trained policy demo.
- `python train/capture_screenshots.py`: regenerate screenshots in `docs/images/`.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and PEP 8-style naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Prefer clear, explicit names for RL concepts (e.g., `obs_dim`, `reward`, `actions`).
- No formatter or linter is configured; keep diffs minimal and readable.

## Testing Guidelines
- No automated test suite is currently present.
- When adding features, include a minimal runnable check (e.g., a new script in `train/` or a reproducible command in `README.md`).
- If you add tests, document how to run them here and in `README.md`.

## Change Log Rule
- When Codex updates any files, add a short summary of changes to `docs/PROJECT_LOG.md`.

## Commit & Pull Request Guidelines
- Recent commits are short, imperative phrases (e.g., “docs”, “screenshots”); no strict convention observed.
- Keep commit subjects concise and scoped to the change.
- PRs should include: a summary, commands run, and screenshots when changing rendering or visuals.

## Security & Configuration Tips
- Training and demo scripts may open windows unless `--headless` is used.
- Checkpoints can be large; keep them in `checkpoints/` and avoid committing large binaries.
