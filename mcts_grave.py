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

from game_engine import Game, P1, P2

# Import shared infrastructure from RAVE (node class, expand, rollout, backprop)
from mcts_rave import (
    RAVENode, _init_moves, expand, rollout, backpropagate,
    C_UCT, RAVE_K, MAX_ROLLOUT_DEPTH, MAX_GAME_MOVES,
)

# ── Constants ──────────────────────────────────────────────────────────────────

GRAVE_THRESHOLD = 50    # minimum N to qualify as amaf reference node
                        # ablate over: {5, 10, 50, 100, 200}


# GRAVE-specific child value 

def grave_child_value(child, parent, c, k, ref_node):
    """
    GRAVE UCT value for `child` at `parent`, using ref_node for AMAF lookup.

      UCT  term: child.W / child.N + exploration  (from parent.player's perspective)
      AMAF term: ref_node.amaf_W[child.move]       (from ref_node.player's perspective)

    ref_node.player == parent.player because both are on the same player's turn
    at the level of the ref_node — same sign convention. ✓
    If ref_node is None (no ancestor has enough visits), fall back to pure UCT.
    """
    if child.N == 0:
        return float('inf')

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
        self.c             = c
        self.k             = k
        self.threshold     = threshold
        self.name          = f"GRAVE({n_simulations},k={k},t={threshold})"

    def choose_move(self, game, state, player):
        return grave_mcts_move(game, state, player,
                               self.n_simulations, self.c, self.k, self.threshold)


# Standalone benchmark 

if __name__ == "__main__":
    from mcts_uct import MCTSAgent, RandomAgent, evaluate
    from mcts_rave import RAVEAgent

    game = Game()

    print("=== GRAVE vs UCT (t=50) ===")
    evaluate(game, GRAVEAgent(500, k=300, threshold=50), MCTSAgent(500), n_games=40)

    print("=== GRAVE vs RAVE (t=50) ===")
    evaluate(game, GRAVEAgent(500, k=300, threshold=50), RAVEAgent(500, k=300), n_games=40)

    print("=== GRAVE threshold ablation vs UCT-500 (20 games each) ===")
    ref = MCTSAgent(500)
    for t in [5, 10, 50, 100, 200]:
        evaluate(game, GRAVEAgent(500, k=300, threshold=t), ref, n_games=20)
