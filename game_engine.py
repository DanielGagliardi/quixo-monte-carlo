SIZE = 5

EMPTY = 0
P1 = 1
P2 = -1


# Pre-compute border positions once at module load time
BORDER_POSITIONS = [
    (r, c) for r in range(SIZE) for c in range(SIZE)
    if r == 0 or r == SIZE-1 or c == 0 or c == SIZE-1
]

def create_board():
    return tuple(tuple(EMPTY for _ in range(SIZE)) for _ in range(SIZE))


def is_border(r, c):
    return r == 0 or r == SIZE - 1 or c == 0 or c == SIZE - 1

def opponent(player):
    return -player  # P1 ↔ P2 since they are +1 and -1



# ---------- Core Mechanics ----------

# push — avoid full list comprehension, copy only the affected row/col
def push(board, r, c, dr, dc, player):
    new = [list(row) for row in board]
    if r == dr:
        row = new[r]
        if dc == 0:
            del row[c]; row.insert(0, player)
        else:
            del row[c]; row.append(player)
    else:
        col = [new[i][c] for i in range(SIZE)]
        if dr == 0:
            del col[r]; col.insert(0, player)
        else:
            del col[r]; col.append(player)
        for i in range(SIZE):
            new[i][c] = col[i]
    return tuple(tuple(row) for row in new)


def next_state(board, move, player):
    r, c, dr, dc = move
    return push(board, r, c, dr, dc, player)


# ---------- Move Generation ----------

def legal_moves(board, player):
    moves = []
    for r, c in BORDER_POSITIONS:           # skip is_border check entirely
        if board[r][c] != EMPTY and board[r][c] != player:
            continue
        if c != 0:        moves.append((r, c, r, 0))
        if c != SIZE - 1: moves.append((r, c, r, SIZE - 1))
        if r != 0:        moves.append((r, c, 0, c))
        if r != SIZE - 1: moves.append((r, c, SIZE - 1, c))
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


MAX_MOVES = 200  # safe upper bound; typical game is 10-50 moves

def is_terminal(board, move_count=None):
    if has_line(board, P1) or has_line(board, P2):
        return True
    if move_count is not None and move_count >= MAX_MOVES:
        return True  # forced draw
    return False



def reward(board, last_mover):
    """
    Returns +1 if last_mover won, -1 if last_mover lost.
    MUST be called with the player who made the last move.
    For the opponent's reward, negate: -reward(board, last_mover).
    """
    opponent = -last_mover
    if has_line(board, opponent): return -1  # formed opponent's line = loss
    if has_line(board, last_mover): return 1
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