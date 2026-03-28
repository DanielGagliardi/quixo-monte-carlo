import math
import random

SIZE = 5

EMPTY = 0
P1 = 1
P2 = -1


# ---------- Game Logic ----------

def create_board():
    return tuple(tuple(EMPTY for _ in range(SIZE)) for _ in range(SIZE))


def is_border(r, c):
    return r == 0 or r == SIZE - 1 or c == 0 or c == SIZE - 1


def push(board, r, c, dr, dc, player):
    new = [list(row) for row in board]

    if r == dr:
        if dc == 0:
            for i in range(c, 0, -1):
                new[r][i] = new[r][i - 1]
            new[r][0] = player
        else:
            for i in range(c, SIZE - 1):
                new[r][i] = new[r][i + 1]
            new[r][SIZE - 1] = player
    else:
        if dr == 0:
            for i in range(r, 0, -1):
                new[i][c] = new[i - 1][c]
            new[0][c] = player
        else:
            for i in range(r, SIZE - 1):
                new[i][c] = new[i + 1][c]
            new[SIZE - 1][c] = player

    return tuple(tuple(row) for row in new)


def next_state(board, move, player):
    r, c, dr, dc = move
    return push(board, r, c, dr, dc, player)


def legal_moves(board, player):
    moves = []

    for r in range(SIZE):
        for c in range(SIZE):
            if not is_border(r, c):
                continue
            if board[r][c] not in (EMPTY, player):
                continue

            if c != 0:
                moves.append((r, c, r, 0))
            if c != SIZE - 1:
                moves.append((r, c, r, SIZE - 1))
            if r != 0:
                moves.append((r, c, 0, c))
            if r != SIZE - 1:
                moves.append((r, c, SIZE - 1, c))

    return moves


def has_line(board, player):
    for i in range(SIZE):
        if all(board[i][j] == player for j in range(SIZE)):
            return True
        if all(board[j][i] == player for j in range(SIZE)):
            return True

    if all(board[i][i] == player for i in range(SIZE)):
        return True
    if all(board[i][SIZE - 1 - i] == player for i in range(SIZE)):
        return True

    return False


def is_terminal(board):
    return has_line(board, P1) or has_line(board, P2)


def reward(board, player):
    opponent = -player
    if has_line(board, opponent):
        return -1
    if has_line(board, player):
        return 1
    return 0


class Game:
    def initial_state(self):
        return create_board()

    def legal_moves(self, state, player):
        return legal_moves(state, player)

    def next_state(self, state, move, player):
        return next_state(state, move, player)

    def is_terminal(self, state):
        return is_terminal(state)

    def reward(self, state, player):
        return reward(state, player)


# ---------- MCTS ----------

class Node:
    def __init__(self, state, player, parent=None):
        self.state = state
        self.player = player
        self.parent = parent

        self.children = {}
        self.untried_moves = None

        self.N = 0
        self.W = 0.0


def ucb1(parent, child, c=1.4):
    if child.N == 0:
        return float("inf")

    Q = child.W / child.N
    Qc = (Q + 1.0) / 2.0

    return Qc + c * math.sqrt(math.log(parent.N) / child.N)


def select(node, game):
    while True:
        if node.untried_moves is None:
            node.untried_moves = game.legal_moves(node.state, node.player)

        if node.untried_moves or game.is_terminal(node.state):
            return node

        node = max(node.children.values(),
                   key=lambda child: ucb1(node, child))


def expand(node, game):
    move = node.untried_moves.pop()
    next_s = game.next_state(node.state, move, node.player)

    child = Node(next_s, -node.player, node)
    node.children[move] = child
    return child


def rollout(state, player, game):
    s = state
    p = player

    while not game.is_terminal(s):
        move = random.choice(game.legal_moves(s, p))
        s = game.next_state(s, move, p)
        p = -p

    return game.reward(s, player)


def backprop(node, result):
    while node is not None:
        node.N += 1
        node.W += result
        result = -result
        node = node.parent


def mcts(state, player, game, iterations=2000):
    root = Node(state, player)

    for _ in range(iterations):
        node = select(root, game)

        if not game.is_terminal(node.state):
            node = expand(node, game)

        result = rollout(node.state, node.player, game)
        backprop(node, result)

    return max(root.children.items(), key=lambda x: x[1].N)[0]


# ---------- Display ----------

def print_board(board):
    symbols = {P1: "X", P2: "O", EMPTY: "."}
    print("  a b c d e")
    for i, row in enumerate(board):
        print(f"{i+1} " + " ".join(symbols[x] for x in row))
    print()


def move_str(move):
    r, c, dr, dc = move
    return f"{chr(c+97)}{r+1}->{chr(dc+97)}{dr+1}"


# ---------- Self-play ----------

def self_play(iterations=4000):
    game = Game()
    state = game.initial_state()
    player = P1
    turn = 0

    while not game.is_terminal(state):
        print(f"Turn {turn}, Player {'X' if player == P1 else 'O'}")
        print_board(state)

        move = mcts(state, player, game, iterations=iterations)
        print("Move:", move_str(move))

        state = game.next_state(state, move, player)
        player = -player
        turn += 1

    print("Final position:")
    print_board(state)

    if has_line(state, P1):
        print("X wins")
    elif has_line(state, P2):
        print("O wins")


if __name__ == "__main__":
    self_play()