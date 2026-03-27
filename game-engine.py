SIZE = 5

EMPTY = 0
P1 = 1
P2 = -1


def create_board():
    return tuple(tuple(EMPTY for _ in range(SIZE)) for _ in range(SIZE))


def is_border(r, c):
    return r == 0 or r == SIZE - 1 or c == 0 or c == SIZE - 1


# ---------- Core Mechanics ----------

def push(board, r, c, dr, dc, player):
    new = [list(row) for row in board]

    if r == dr:
        if dc == 0:  # push from left
            for i in range(c, 0, -1):
                new[r][i] = new[r][i - 1]
            new[r][0] = player
        else:  # push from right
            for i in range(c, SIZE - 1):
                new[r][i] = new[r][i + 1]
            new[r][SIZE - 1] = player

    else:
        if dr == 0:  # push from top
            for i in range(r, 0, -1):
                new[i][c] = new[i - 1][c]
            new[0][c] = player
        else:  # push from bottom
            for i in range(r, SIZE - 1):
                new[i][c] = new[i + 1][c]
            new[SIZE - 1][c] = player

    return tuple(tuple(row) for row in new)


def next_state(board, move, player):
    r, c, dr, dc = move
    return push(board, r, c, dr, dc, player)


# ---------- Move Generation ----------

def legal_moves(board, player):
    moves = []

    for r in range(SIZE):
        for c in range(SIZE):
            if not is_border(r, c):
                continue

            if board[r][c] not in (EMPTY, player):
                continue

            # same row → left/right edges
            if c != 0:
                moves.append((r, c, r, 0))
            if c != SIZE - 1:
                moves.append((r, c, r, SIZE - 1))

            # same column → top/bottom
            if r != 0:
                moves.append((r, c, 0, c))
            if r != SIZE - 1:
                moves.append((r, c, SIZE - 1, c))

    return moves


# ---------- Win Conditions ----------

def has_line(board, player):
    # rows
    for r in range(SIZE):
        if all(board[r][c] == player for c in range(SIZE)):
            return True

    # columns
    for c in range(SIZE):
        if all(board[r][c] == player for r in range(SIZE)):
            return True

    # diagonals
    if all(board[i][i] == player for i in range(SIZE)):
        return True

    if all(board[i][SIZE - 1 - i] == player for i in range(SIZE)):
        return True

    return False


def is_terminal(board):
    return has_line(board, P1) or has_line(board, P2)


def reward(board, player):
    opponent = -player

    # precedence rule: forming opponent line = loss
    if has_line(board, opponent):
        return -1

    if has_line(board, player):
        return 1

    return 0


# ---------- Game Wrapper ----------

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


# ---------- Utilities (optional, not used by MCTS) ----------

def print_board(board):
    symbols = {P1: "X", P2: "O", EMPTY: "."}
    print("  a b c d e")
    for i, row in enumerate(board):
        print(f"{i+1} " + " ".join(symbols[x] for x in row))
    print()


def move_to_str(move):
    r, c, dr, dc = move
    return f"{chr(c + 97)}{r+1}{chr(dc + 97)}{dr+1}"