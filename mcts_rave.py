# mcts_rave.py
"""
MCTS with RAVE (Rapid Action Value Estimation) / AMAF statistics.

Builds directly on game_engine.py and shares all conventions with mcts_uct.py.

Key change vs UCT:
  - Each node stores AMAF stats: for each move m, how often node.player
    played m in rollouts through this node, and the average outcome.
  - Selection blends UCT and AMAF via β-schedule:
        Q_RAVE = (1-β)*Q_UCT + β*Q_AMAF
        β = sqrt(k / (3*N + k))   →  0 as N grows (UCT dominates asymptotically)
  - Backprop updates both W/N (UCT) and amaf_W/amaf_N (RAVE) at every node.

Perspective convention (identical to mcts_uct.py):
  node.W          — accumulated value from node.parent.player's perspective
  node.amaf_W[m]  — accumulated value of move m from node.player's perspective
                    (parent accesses parent.amaf_W[child.move] which is from
                     parent.player's perspective — same sign as child.W/child.N ✓)
"""

import math
import random
import time

from game_engine import Game, P1, P2, SIZE

# Constants 

C_UCT             = math.sqrt(2)
RAVE_K            = 300        # β schedule: larger k → trust AMAF longer
                               # tune over {100, 300, 1000, 3000}
MAX_ROLLOUT_DEPTH = 200
MAX_GAME_MOVES    = 400


# ── Node ───────────────────────────────────────────────────────────────────────

class RAVENode:
    """
    UCT node extended with per-move AMAF statistics.

    amaf_N[m] — number of times node.player played move m in rollouts through here
    amaf_W[m] — cumulative outcome of those rollouts, from node.player's perspective
    """
    __slots__ = ('state', 'player', 'parent', 'move',
                 'children', 'untried_moves', 'N', 'W',
                 'amaf_N', 'amaf_W')

    def __init__(self, state, player, parent=None, move=None):
        self.state         = state
        self.player        = player
        self.parent        = parent
        self.move          = move
        self.children      = []
        self.untried_moves = None
        self.N             = 0
        self.W             = 0.0
        self.amaf_N        = {}   # move → int
        self.amaf_W        = {}   # move → float

    def is_fully_expanded(self):
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def rave_value(self, c, k):
        """
        β-blended value from parent's perspective.
        Called on a CHILD node; reads amaf from self.parent.
        """
        if self.N == 0:
            return float('inf')

        parent = self.parent
        q_uct  = self.W / self.N + c * math.sqrt(math.log(parent.N) / self.N)

        amaf_n = parent.amaf_N.get(self.move, 0)
        if amaf_n == 0:
            return q_uct                        # no AMAF data yet → pure UCT

        beta   = math.sqrt(k / (3.0 * parent.N + k))
        q_amaf = parent.amaf_W[self.move] / amaf_n
        return (1.0 - beta) * q_uct + beta * q_amaf

    def best_child(self, c, k):
        return max(self.children, key=lambda ch: ch.rave_value(c, k))

    def most_visited_child(self):
        return max(self.children, key=lambda ch: ch.N)

    def win_rate(self):
        return self.W / self.N if self.N > 0 else 0.0


# Helpers 

def _init_moves(node, game):
    if node.untried_moves is None:
        node.untried_moves = game.legal_moves(node.state, node.player)


# Four MCTS Phases 

def select(root, game, c, k):
    """
    Descend tree via RAVE-UCT.
    Returns (leaf, tree_moves) where tree_moves[player] = set of moves
    played by that player during tree traversal (used for AMAF updates).
    """
    node       = root
    tree_moves = {P1: set(), P2: set()}

    while not game.is_terminal(node.state):
        _init_moves(node, game)
        if not node.is_fully_expanded():
            return node, tree_moves
        best = node.best_child(c, k)
        tree_moves[node.player].add(best.move)
        node = best

    return node, tree_moves


def expand(node, game):
    """
    Add one child for a random untried move.
    Returns (child_node, move_played).
    """
    _init_moves(node, game)
    idx   = random.randrange(len(node.untried_moves))
    move  = node.untried_moves.pop(idx)
    child = RAVENode(
        state  = game.next_state(node.state, move, node.player),
        player = -node.player,
        parent = node,
        move   = move,
    )
    node.children.append(child)
    return child, move


def rollout(node, game):
    """
    Random playout from node. Returns (value, result_p1, sim_moves).

      value      — from node.parent's perspective  (for W backprop, same as UCT)
      result_p1  — ∈ {-1, 0, +1} from P1's perspective  (for AMAF sign computation)
      sim_moves  — {P1: set_of_moves, P2: set_of_moves} played during simulation
    """
    state     = node.state
    player    = node.player
    depth     = 0
    sim_moves = {P1: set(), P2: set()}

    while not game.is_terminal(state) and depth < MAX_ROLLOUT_DEPTH:
        moves  = game.legal_moves(state, player)
        move   = random.choice(moves)
        sim_moves[player].add(move)
        state  = game.next_state(state, move, player)
        player = -player
        depth += 1

    if depth >= MAX_ROLLOUT_DEPTH:
        return 0.0, 0, sim_moves                     # draw

    last_mover = -player
    result     = game.reward(state, last_mover)      # ±1 from last_mover's view
    result_p1  = result * last_mover                 # ±1 from P1's view

    # Convert to node.parent's perspective (identical logic to mcts_uct.py)
    value = -result if last_mover == node.player else result
    return value, result_p1, sim_moves


def backpropagate(node, value, result_p1, all_moves):
    """
    Walk up tree updating N, W (UCT) and amaf_N, amaf_W (RAVE).

    AMAF update at node n:
      For every move m that n.player played (tree + simulation),
      record the outcome from n.player's perspective in n.amaf_W[m].
      The parent then reads n.amaf_W[m] via parent.amaf_W[child.move],
      which is from parent.player's perspective — matching child.W/child.N. ✓
    """
    while node is not None:
        node.N += 1
        node.W += value

        # v_player: outcome from node.player's perspective
        v_player = result_p1 * node.player       # +1 if node.player won
        for move in all_moves[node.player]:
            node.amaf_N[move] = node.amaf_N.get(move, 0)   + 1
            node.amaf_W[move] = node.amaf_W.get(move, 0.0) + v_player

        value = -value
        node  = node.parent


# Main Entry Point 

def rave_mcts_move(game, state, player, n_simulations=500, c=C_UCT, k=RAVE_K):
    """Run RAVE-MCTS and return the best move for `player` at `state`."""
    root = RAVENode(state, player)

    for _ in range(n_simulations):

        # Selection
        leaf, tree_moves = select(root, game, c, k)

        # Expansion — also record the expansion move in tree_moves
        if not game.is_terminal(leaf.state):
            leaf, exp_move = expand(leaf, game)
            tree_moves[leaf.parent.player].add(exp_move)

        # Simulation
        value, result_p1, sim_moves = rollout(leaf, game)

        # Merge tree-traversal moves + simulation moves for AMAF
        all_moves = {
            P1: tree_moves[P1] | sim_moves[P1],
            P2: tree_moves[P2] | sim_moves[P2],
        }

        # Backpropagation
        backpropagate(leaf, value, result_p1, all_moves)

    return root.most_visited_child().move


# Agent 

class RAVEAgent:
    """RAVE-MCTS agent with configurable simulation budget, UCT constant, and k."""

    def __init__(self, n_simulations=500, c=C_UCT, k=RAVE_K):
        self.n_simulations = n_simulations
        self.c             = c
        self.k             = k
        self.name          = f"RAVE({n_simulations},k={k})"

    def choose_move(self, game, state, player):
        return rave_mcts_move(game, state, player,
                              self.n_simulations, self.c, self.k)


# Standalone benchmark 

if __name__ == "__main__":
    from mcts_uct import MCTSAgent, RandomAgent, GreedyAgent, evaluate

    game = Game()

    print("=== RAVE vs UCT (same budget) ===")
    evaluate(game, RAVEAgent(500, k=300), MCTSAgent(500), n_games=40)

    print("=== RAVE k ablation vs UCT-500 (20 games each) ===")
    ref = MCTSAgent(500)
    for k_val in [100, 300, 1000, 3000]:
        evaluate(game, RAVEAgent(500, k=k_val), ref, n_games=20)
