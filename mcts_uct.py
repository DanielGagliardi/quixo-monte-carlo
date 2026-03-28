# Vanilla MCTS with UCT selection policy
# Includes: RandomAgent, GreedyAgent, MCTSAgent, evaluation harness

import math
import random
import time
from game_engine import Game, P1, P2, SIZE

# Constants 

C_UCT            = math.sqrt(2)   # exploration constant  (tune: try 0.5, 1.0, √2)
MAX_ROLLOUT_DEPTH = 200            # max moves in a single simulation rollout
MAX_GAME_MOVES    = 400            # safety cap for full agent-vs-agent games


# NODE

class Node:
    """
    W stores accumulated value from the perspective of the PARENT's player
    (i.e., the player who chose to move to this node).
    This means selection at any node is simply argmax(W/N + exploration)
    with no sign flip needed.
    """
    __slots__ = ('state', 'player', 'parent', 'move',
                 'children', 'untried_moves', 'N', 'W')

    def __init__(self, state, player, parent=None, move=None):
        self.state         = state
        self.player        = player    # player whose turn it is at this node
        self.parent        = parent
        self.move          = move      # move that led to this state (None for root)
        self.children      = []
        self.untried_moves = None      # lazily initialised on first visit
        self.N             = 0         # visit count
        self.W             = 0.0       # total value from parent's perspective

    # helpers

    def is_fully_expanded(self):
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def uct_value(self, c):
        if self.N == 0:
            return float('inf')
        return self.W / self.N + c * math.sqrt(math.log(self.parent.N) / self.N)

    def best_child(self, c):
        """UCT child — used during tree traversal."""
        return max(self.children, key=lambda ch: ch.uct_value(c))

    def most_visited_child(self):
        """Robust final move selection: pick highest-N child (less variance than UCT)."""
        return max(self.children, key=lambda ch: ch.N)

    def win_rate(self):
        return self.W / self.N if self.N > 0 else 0.0


# FOUR MCTS PHASES

def _init_moves(node, game):
    """Lazily populate untried_moves on first visit."""
    if node.untried_moves is None:
        node.untried_moves = game.legal_moves(node.state, node.player)


# Selection 

def select(root, game, c):
    """Descend the tree via UCT until we reach a node that is not fully expanded
    or a terminal state."""
    node = root
    while not game.is_terminal(node.state):
        _init_moves(node, game)
        if not node.is_fully_expanded():
            return node          # stop here: this node needs expansion
        node = node.best_child(c)
    return node                  # terminal node — will skip expansion


# Expansion

def expand(node, game):
    """Add one new child for a randomly chosen untried move."""
    _init_moves(node, game)
    idx  = random.randrange(len(node.untried_moves))
    move = node.untried_moves.pop(idx)

    next_state = game.next_state(node.state, move, node.player)
    child = Node(next_state, -node.player, parent=node, move=move)
    node.children.append(child)
    return child


# Simulation (rollout) 

def rollout(node, game):
    """
    Random playout from node's state.

    Returns value from the PARENT's perspective (= -node.player perspective),
    so that backpropagate() can simply add it to node.W and flip sign going up.

    Derivation:
      - Rollout starts with node.player moving first.
      - After `depth` half-moves, last_mover = node.player  if depth is odd
                                              = -node.player if depth is even
      - game.reward(state, last_mover) is +1/-1 from last_mover's view.
      - We want the value from -node.player's view (the parent's player):
          if last_mover == node.player  → negate  (opposite perspectives)
          if last_mover == -node.player → keep as-is
    """
    state  = node.state
    player = node.player
    depth  = 0

    while not game.is_terminal(state) and depth < MAX_ROLLOUT_DEPTH:
        moves = game.legal_moves(state, player)
        state  = game.next_state(state, random.choice(moves), player)
        player = -player
        depth += 1

    if depth >= MAX_ROLLOUT_DEPTH:
        return 0.0   # treat as draw

    last_mover = -player                          # who made the final move
    result = game.reward(state, last_mover)       # +1/-1 from last_mover's view

    # Convert to parent's perspective (= -node.player)
    if last_mover == node.player:
        return -result   # node.player and -node.player are opponents → flip
    else:
        return result    # last_mover is already -node.player → keep


# Backpropagation 

def backpropagate(node, value):
    """
    Walk up from node to root, updating N and W.
    `value` starts as the result from node.parent's perspective.
    At each level we flip sign because parent and grandparent are opponents.
    """
    while node is not None:
        node.N += 1
        node.W += value
        value   = -value
        node    = node.parent



#  MAIN MCTS ENTRY POINT

def mcts_move(game, state, player, n_simulations=500, c=C_UCT):
    """Run MCTS and return the best move for `player` at `state`."""
    root = Node(state, player)

    for _ in range(n_simulations):
        leaf  = select(root, game, c)
        if not game.is_terminal(leaf.state):
            leaf = expand(leaf, game)
        value = rollout(leaf, game)
        backpropagate(leaf, value)

    return root.most_visited_child().move


#  AGENTS

class RandomAgent:
    """Uniform random move selection — weakest baseline."""
    name = "Random"

    def choose_move(self, game, state, player):
        return random.choice(game.legal_moves(state, player))


class GreedyAgent:
    """
    One-ply heuristic: pick the move that maximises your longest line,
    breaking ties by minimising the opponent's longest line.
    """
    name = "Greedy"

    @staticmethod
    def _max_line(board, player):
        best = 0
        for r in range(SIZE):
            best = max(best, sum(1 for c in range(SIZE) if board[r][c] == player))
        for c in range(SIZE):
            best = max(best, sum(1 for r in range(SIZE) if board[r][c] == player))
        best = max(best, sum(1 for i in range(SIZE) if board[i][i]         == player))
        best = max(best, sum(1 for i in range(SIZE) if board[i][SIZE-1-i]  == player))
        return best

    def choose_move(self, game, state, player):
        opponent = -player
        best_move, best_score = None, (-999, 999)
        for move in game.legal_moves(state, player):
            ns    = game.next_state(state, move, player)
            score = (self._max_line(ns, player), -self._max_line(ns, opponent))
            if score > best_score:
                best_score, best_move = score, move
        return best_move


class MCTSAgent:
    """UCT MCTS agent with configurable simulation budget and exploration constant."""

    def __init__(self, n_simulations=500, c=C_UCT):
        self.n_simulations = n_simulations
        self.c             = c
        self.name          = f"MCTS({n_simulations})"

    def choose_move(self, game, state, player):
        return mcts_move(game, state, player, self.n_simulations, self.c)


#  EVALUATION HARNESS

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
        return 0   # draw by move limit

    last_mover = -player
    # reward(state, last_mover) is ±1 from last_mover's view.
    # Multiply by last_mover (+1 or -1) to convert to P1's view.
    return game.reward(state, last_mover) * last_mover


def evaluate(game, agent1, agent2, n_games=100, verbose=False):
    """
    Play n_games: half with agent1 as P1, half with sides swapped.
    Prints a result table and returns {'W': wins, 'D': draws, 'L': losses}.
    """
    results    = {'W': 0, 'D': 0, 'L': 0}
    half       = n_games // 2
    t0         = time.time()

    for i in range(half):
        r = play_game(game, agent1, agent2, verbose)
        if r == 1:  results['W'] += 1
        elif r == 0: results['D'] += 1
        else:        results['L'] += 1
        print(f"  game {i+1:3d}/{half} (a1=P1)  outcome: {r:+d}", end='\r')

    for i in range(half):
        r = play_game(game, agent2, agent1, verbose)   # sides swapped
        # r is from agent2/P1's view → negate for agent1
        r = -r
        if r == 1:  results['W'] += 1
        elif r == 0: results['D'] += 1
        else:        results['L'] += 1
        print(f"  game {i+1:3d}/{half} (a1=P2)  outcome: {r:+d}", end='\r')

    elapsed = time.time() - t0
    W, D, L = results['W'], results['D'], results['L']
    print(f"\n{'─'*50}")
    print(f"  {agent1.name:20s}  vs  {agent2.name}")
    print(f"  Games : {n_games}   ({elapsed:.1f}s)")
    print(f"  Wins  : {W:3d}  ({100*W/n_games:5.1f}%)")
    print(f"  Draws : {D:3d}  ({100*D/n_games:5.1f}%)")
    print(f"  Losses: {L:3d}  ({100*L/n_games:5.1f}%)")
    print(f"{'─'*50}\n")
    return results


#  MAIN  — benchmark suite

if __name__ == "__main__":
    game = Game()

    random_agent = RandomAgent()
    greedy_agent = GreedyAgent()
    mcts_100     = MCTSAgent(n_simulations=100)
    mcts_500     = MCTSAgent(n_simulations=500)
    mcts_1000    = MCTSAgent(n_simulations=1000)

    # Tier 1: beat the baselines 
    print("=== MCTS vs Random ===")
    evaluate(game, mcts_100,  random_agent, n_games=50)
    evaluate(game, mcts_500,  random_agent, n_games=50)

    print("=== MCTS vs Greedy ===")
    evaluate(game, mcts_100,  greedy_agent, n_games=50)
    evaluate(game, mcts_500,  greedy_agent, n_games=50)

    # Tier 2: budget sensitivity 
    print("=== Budget sensitivity ===")
    evaluate(game, mcts_500,  mcts_100,  n_games=40)
    evaluate(game, mcts_1000, mcts_500,  n_games=20)

    #  Tier 3: single verbose game (sanity check) 
    print("=== One verbose game: MCTS(200) vs Random ===")
    mcts_200 = MCTSAgent(n_simulations=200)
    play_game(game, mcts_200, random_agent, verbose=True)
