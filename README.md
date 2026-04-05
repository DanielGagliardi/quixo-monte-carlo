# Quixo Monte Carlo

Monte Carlo Tree Search research project for the board game Quixo (Gigamic).

This repository includes:
- A fast Quixo engine.
- Baseline UCT MCTS.
- RAVE and GRAVE variants.
- Evaluation harnesses with crash-safe checkpoints and resumable runs.
- Hyperparameter search scripts.
- Experimental AlphaZero-style and PPO-related workspaces.

![Quixo Demo](visualisations/quixo_multi_loop.gif)

## Table of Contents

- [Project Goals](#project-goals)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Algorithms Included](#algorithms-included)
- [Running Experiments](#running-experiments)
- [Logs and Checkpoints](#logs-and-checkpoints)
- [Alpha and PPO Folders](#alpha-and-ppo-folders)
- [Notes on Reproducibility](#notes-on-reproducibility)

## Project Goals

The project is designed to compare search-based Quixo agents under a controlled evaluation pipeline.

Main objectives:
- Implement strong, clear baselines.
- Measure the effect of RAVE and GRAVE enhancements.
- Run long experiments safely with resume support.
- Explore budget scaling and hyperparameter sensitivity.

## Repository Structure

Top-level files and folders:

- `game_engine.py`: Core game logic (state transitions, legal moves, terminal checks, reward).
- `game_terminal.py`: Human vs human text-mode Quixo.
- `game_gui.py`: Interactive Pygame board for manual play.
- `mcts_uct.py`: Vanilla MCTS with UCT policy, agents, and evaluation harness.
- `mcts_rave.py`: RAVE MCTS with AMAF statistics and blended selection.
- `mcts_grave.py`: GRAVE MCTS using ancestor AMAF references.
- `benchmark_search.py`: Multi-stage hyperparameter benchmark and leaderboard.
- `self_play_draw_test.py`: Self-play diagnostic script for draw-rate analysis.
- `alpha_quixo.py`: AlphaZero-style network + PUCT-style search prototype.
- `playground.ipynb`: Notebook scratchpad.
- `visualisations/`: Visualization notebooks and generated figures.
- `alpha-quixo/`: Alpha-related notebooks, checkpoints, and playtest scripts.
- `PPO-quixo/`: PPO notebooks/logs and related experimental files.
- `logs/`: Per-run log files.
- `checkpoints/`: JSON checkpoints for resumable evaluations.

## Requirements

Python 3.9+ recommended.

Minimum dependencies for core MCTS scripts:
- Standard library only (for `game_engine.py`, `mcts_uct.py`, `mcts_rave.py`, `mcts_grave.py`, `benchmark_search.py`).

Additional dependencies for optional modules:
- `pygame` for `game_gui.py`.
- `numpy` and `torch` for `alpha_quixo.py` and Alpha-related experiments.
- Notebook and plotting stack for `.ipynb` workflows (typically `jupyter`, `matplotlib`, `pandas`, `networkx` depending on notebook cell usage).

Suggested setup:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install pygame numpy torch jupyter matplotlib pandas networkx
```

If you only want core MCTS benchmarking, no extra package install is required.

## Quick Start

Run commands from the repository root.

1. Play in terminal:

```bash
python game_terminal.py
```

2. Play in GUI:

```bash
python game_gui.py
```

3. Run baseline UCT benchmarks:

```bash
python mcts_uct.py
```

4. Run full staged hyperparameter search:

```bash
python benchmark_search.py
```

5. Run GRAVE self-play draw test:

```bash
python self_play_draw_test.py
```

## Algorithms Included

### UCT (Vanilla MCTS)

- Tree policy: UCT selection.
- Default exploration constant is currently set to `C_UCT = 0` in the source files.
- Includes baseline agents:
	- `RandomAgent`
	- `GreedyAgent`
	- `MCTSAgent`

### RAVE (Rapid Action Value Estimation)

- Adds AMAF statistics per node.
- Blends UCT and AMAF values using a beta schedule.
- Controlled by parameter `k` (`RAVE_K` in code).

### GRAVE (Generalized RAVE)

- Uses AMAF data from nearest well-visited ancestor to reduce noise.
- Adds threshold parameter for reference ancestor selection.
- Shares rollout and backprop mechanics with RAVE.

## Running Experiments

### Single Matchup Pattern

All three search modules expose a similar `evaluate(...)` function:
- Plays half games with each side assignment.
- Logs each game immediately.
- Saves checkpoints after every game.
- Supports resume for interrupted runs.

### Hyperparameter Search (`benchmark_search.py`)

The benchmark script executes multiple stages, including:
- Candidate confirmation.
- GRAVE threshold sweep.
- GRAVE `k x threshold` grid.
- RAVE high-`k` sweep.
- Exploration constant ablation.
- Budget scaling to 1000 simulations.
- Championship round-robin with automatic champion selection.

A leaderboard is printed after stages and at the end.

## Logs and Checkpoints

Output folders are created automatically:

- `logs/`: human-readable run logs (`<run_id>.log`).
- `checkpoints/`: crash-safe JSON checkpoints (`<run_id>.json`).

Checkpoint writing uses atomic replace (temporary file then rename), so partial writes are avoided even on interruptions.

Resume behavior summary:
- If a run is unfinished, rerunning with same `run_id` continues from last completed game.
- If a run is already finished, scripts may create a timestamped new run ID.

## Alpha and PPO Folders

These folders contain ongoing/experimental work:

- `alpha-quixo/`: notebooks, model checkpoints (`model_*.pt`), optimizer checkpoints, and playtest scripts.
- `PPO-quixo/`: PPO notebooks and training logs.

Treat these as research workspaces that may evolve independently of the core MCTS benchmark pipeline.

## Notes on Reproducibility

- The engine uses deterministic rules and explicit side swapping in evaluation.
- Some components rely on random rollout and random move selection.
- For strict reproducibility, set random seeds in all relevant modules (`random`, `numpy`, and `torch` where applicable).
- Keep `run_id` stable to resume exactly the same experiment series.

## License

This project is distributed under the license provided in `LICENSE`.
