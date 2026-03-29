"""
MCTS with GRAVE (Generalized RAVE).

GRAVE differs from RAVE only in how AMAF stats are USED during selection:
- Standard RAVE: use amaf stats from the CURRENT node (may be noisy if N is small)
- GRAVE: use amaf stats from the NEAREST ANCESTOR with N >= ref_threshold

This reduces noise at shallow-visited nodes by borrowing statistics from
a well-visited ancestor whose amaf tables are more reliable.

Backpropagation is IDENTICAL to RAVE — amaf is still updated at every node.
The threshold is a single ablation parameter; typical values: 10, 50, 200.

Reference: Cazenave (2011) "Generalized Rapid Action Value Estimation"
"""

import math
import random
import time
import json
import os
import logging
from datetime import datetime
from typing import Optional

from game_engine import Game, P1, P2

# Import shared RAVE infrastructure (node class, phases, constants)
from mcts_rave import (
    RAVENode, _init_moves, expand, rollout, backpropagate,
    C_UCT, RAVE_K, MAX_ROLLOUT_DEPTH, MAX_GAME_MOVES,
)

# Constants 

GRAVE_THRESHOLD = 50   # minimum N to qualify as amaf reference node
                       # ablate over: {5, 10, 50, 100, 200}

LOGS_DIR        = "logs"
CHECKPOINTS_DIR = "checkpoints"

# Logging / Checkpointing Helpers 

os.makedirs(LOGS_DIR,        exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def _checkpoint_path(run_id: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"{run_id}.json")


def _log_path(run_id: str) -> str:
    return os.path.join(LOGS_DIR, f"{run_id}.log")


def _setup_run_logger(run_id: str) -> logging.Logger:
    """Return a Logger that writes to both stdout and a per-run .log file."""
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(_log_path(run_id), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def _save_checkpoint(run_id: str, data: dict) -> None:
    """Atomically write checkpoint to disk (tmp → rename, crash-safe)."""
    path = _checkpoint_path(run_id)
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _load_checkpoint(run_id: str) -> Optional[dict]:
    path = _checkpoint_path(run_id)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# GRAVE-specific child value 

def grave_child_value(child, parent, c, k, ref_node):
    """
    GRAVE UCT value for `child` at `parent`, using ref_node for AMAF lookup.

    UCT term:  child.W / child.N + exploration  (from parent.player's perspective)
    AMAF term: ref_node.amaf_W[child.move]       (from ref_node.player's perspective)

    ref_node.player == parent.player because both are on the same player's turn
    at the level of the ref_node — same sign convention. ✓
    If ref_node is None (no ancestor has enough visits), fall back to pure UCT.
    """
    if child.N == 0:
        return float("inf")

    q_uct = child.W / child.N + c * math.sqrt(math.log(parent.N) / child.N)

    if ref_node is None:
        return q_uct

    amaf_n = ref_node.amaf_N.get(child.move, 0)
    if amaf_n == 0:
        return q_uct

    beta   = math.sqrt(k / (3.0 * parent.N + k))
    q_amaf = ref_node.amaf_W[child.move] / amaf_n
    return (1.0 - beta) * q_uct + beta * q_amaf


# GRAVE Selection 

def grave_select(root, game, c, k, threshold):
    """
    Descend tree via GRAVE-UCT.
    Maintains ref_node = closest ancestor (inclusive) with N >= threshold.
    Returns (leaf, tree_moves, ref_node).
    """
    node       = root
    ref_node   = root if root.N >= threshold else None
    tree_moves = {P1: set(), P2: set()}

    while not game.is_terminal(node.state):
        _init_moves(node, game)
        if not node.is_fully_expanded():
            return node, tree_moves, ref_node

        # Update reference before selecting child (so children use this node
        # as ref if it qualifies)
        if node.N >= threshold:
            ref_node = node

        best = max(node.children,
                   key=lambda ch: grave_child_value(ch, node, c, k, ref_node))
        tree_moves[node.player].add(best.move)
        node = best

    return node, tree_moves, ref_node


# Main Entry Point 

def grave_mcts_move(game, state, player,
                    n_simulations=500, c=C_UCT, k=RAVE_K, threshold=GRAVE_THRESHOLD):
    """Run GRAVE-MCTS and return the best move for `player` at `state`."""
    root = RAVENode(state, player)

    for _ in range(n_simulations):

        # Selection (GRAVE variant — tracks ref_node)
        leaf, tree_moves, ref_node = grave_select(root, game, c, k, threshold)

        # Expansion
        if not game.is_terminal(leaf.state):
            leaf, exp_move = expand(leaf, game)
            tree_moves[leaf.parent.player].add(exp_move)

        # Simulation (identical to RAVE)
        value, result_p1, sim_moves = rollout(leaf, game)

        # Merge moves
        all_moves = {
            P1: tree_moves[P1] | sim_moves[P1],
            P2: tree_moves[P2] | sim_moves[P2],
        }

        # Backpropagation (identical to RAVE — amaf updated at every node)
        backpropagate(leaf, value, result_p1, all_moves)

    return root.most_visited_child().move


# Agent 

class GRAVEAgent:
    """GRAVE-MCTS agent. Key ablation parameter: threshold."""

    def __init__(self, n_simulations=500, c=C_UCT, k=RAVE_K, threshold=GRAVE_THRESHOLD):
        self.n_simulations = n_simulations
        self.c         = c
        self.k         = k
        self.threshold = threshold
        self.name      = f"GRAVE({n_simulations},k={k},t={threshold})"

    def choose_move(self, game, state, player):
        return grave_mcts_move(game, state, player,
                               self.n_simulations, self.c, self.k, self.threshold)


# Evaluation Harness 

def play_game(game, agent1, agent2, verbose=False):
    """
    agent1 plays as P1, agent2 plays as P2.
    Returns +1 if agent1 wins, -1 if agent2 wins, 0 for draw.
    """
    state      = game.initial_state()
    agents     = {P1: agent1, P2: agent2}
    player     = P1
    move_count = 0

    while not game.is_terminal(state) and move_count < MAX_GAME_MOVES:
        move   = agents[player].choose_move(game, state, player)
        state  = game.next_state(state, move, player)
        player = -player
        move_count += 1
        if verbose:
            from game_engine import print_board, move_to_str
            print(f"Move {move_count}: {move_to_str(move)}")
            print_board(state)

    if move_count >= MAX_GAME_MOVES:
        return 0

    last_mover = -player
    return game.reward(state, last_mover) * last_mover


def evaluate(
    game,
    agent1,
    agent2,
    n_games: int = 100,
    verbose: bool = False,
    run_id: Optional[str] = None,
    resume: bool = True,
) -> dict:
    """
    Play n_games: half with agent1 as P1, half with sides swapped.
    Results are logged to logs/<run_id>.log and checkpointed to
    checkpoints/<run_id>.json after every single game.

    Parameters
    ----------
    run_id : str | None
        Stable identifier for this matchup.
        Defaults to  "<agent1>_vs_<agent2>_<n_games>g"  (no timestamp),
        so the same run is naturally resumed across script restarts.
        Pass an explicit run_id to manage multiple independent runs of the
        same matchup concurrently.
    resume : bool
        If True (default) and an *unfinished* checkpoint exists for run_id,
        the run continues from where it stopped.
        If False, or if the previous run already finished, a fresh
        timestamped run_id is created automatically.

    Returns
    -------
    dict  {'W': wins, 'D': draws, 'L': losses}
    """
    # ── Resolve run_id and decide whether to resume ──────────────────────────
    base_id    = run_id or f"{agent1.name}_vs_{agent2.name}_{n_games}g"
    checkpoint = _load_checkpoint(base_id) if resume else None

    if checkpoint and checkpoint.get("finished"):
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_id  = f"{base_id}_{ts}"
        checkpoint = None
    else:
        actual_id = base_id

    logger = _setup_run_logger(actual_id)

    # Restore or initialise state 
    if checkpoint:
        outcomes   = checkpoint["outcomes"]
        start_game = checkpoint["completed_games"]
        logger.info(
            f"Resuming '{actual_id}' — "
            f"{start_game}/{n_games} games already done."
        )
    else:
        outcomes   = []
        start_game = 0
        logger.info(
            f"Starting '{actual_id}' — "
            f"{n_games} games  ({agent1.name} vs {agent2.name})"
        )

    half = n_games // 2
    t0   = time.time()

    # Inner helper: run one game, log it immediately, checkpoint 
    def _run_and_save(global_idx: int, a1, a2, label: str) -> int:
        r      = play_game(game, a1, a2, verbose)
        result = r if label == "a1=P1" else -r
        outcomes.append(result)
        sym    = {1: "W", 0: "D", -1: "L"}[result]
        logger.info(
            f"  [{sym}] game {global_idx+1:3d}/{n_games}  "
            f"({label})  raw={r:+d}  agent1_result={result:+d}"
        )
        _save_checkpoint(actual_id, {
            "run_id":          actual_id,
            "agent1":          agent1.name,
            "agent2":          agent2.name,
            "n_games":         n_games,
            "completed_games": len(outcomes),
            "outcomes":        outcomes,
            "finished":        False,
        })
        return result

    # agent1 as P1 
    for i in range(half):
        if i < start_game:
            continue
        _run_and_save(i, agent1, agent2, "a1=P1")
        print(f"  game {i+1:3d}/{half} (a1=P1)", end="\r")

    # agent1 as P2 (sides swapped) 
    for i in range(half):
        global_idx = half + i
        if global_idx < start_game:
            continue
        _run_and_save(global_idx, agent2, agent1, "a1=P2")
        print(f"  game {i+1:3d}/{half} (a1=P2)", end="\r")

    # Tally and report 
    results = {"W": outcomes.count(1), "D": outcomes.count(0), "L": outcomes.count(-1)}
    elapsed = time.time() - t0
    W, D, L = results["W"], results["D"], results["L"]
    n_done  = len(outcomes)

    summary = (
        f"\n{'─'*50}\n"
        f"  {agent1.name:20s} vs {agent2.name}\n"
        f"  Games  : {n_done} / {n_games}  ({elapsed:.1f}s)\n"
        f"  Wins   : {W:3d} ({100*W/n_done:5.1f}%)\n"
        f"  Draws  : {D:3d} ({100*D/n_done:5.1f}%)\n"
        f"  Losses : {L:3d} ({100*L/n_done:5.1f}%)\n"
        f"{'─'*50}"
    )
    logger.info(summary)

    _save_checkpoint(actual_id, {
        "run_id":          actual_id,
        "agent1":          agent1.name,
        "agent2":          agent2.name,
        "n_games":         n_games,
        "completed_games": n_done,
        "outcomes":        outcomes,
        "finished":        True,
        "W": W, "D": D, "L": L,
    })

    return results


# Standalone Benchmark 

if __name__ == "__main__":
    from mcts_uct import MCTSAgent, RandomAgent
    from mcts_rave import RAVEAgent

    game = Game()

    print("=== GRAVE vs UCT (t=50) ===")
    evaluate(game, GRAVEAgent(500, k=300, threshold=50), MCTSAgent(500), n_games=100)

    print("=== GRAVE vs RAVE (t=50) ===")
    evaluate(game, GRAVEAgent(500, k=300, threshold=50), RAVEAgent(500, k=300), n_games=100)

    print("=== GRAVE threshold ablation vs UCT-500 (50 games each) ===")
    ref = MCTSAgent(500)
    for t in [5, 10, 50, 100, 200]:
        evaluate(game, GRAVEAgent(500, k=300, threshold=t), ref, n_games=50)