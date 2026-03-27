SIZE = 5
EMPTY = "."

def create_board():
    return [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]

def print_board(board):
    print("  a b c d e")
    for i, row in enumerate(board):
        print(f"{i+1} " + " ".join(row))
    print()

def coord_to_idx(coord):
    col = ord(coord[0]) - ord('a')
    row = int(coord[1]) - 1
    return row, col

def is_border(r, c):
    return r == 0 or r == SIZE-1 or c == 0 or c == SIZE-1

def parse_move(move):
    if len(move) != 4:
        raise ValueError("Format must be like a2a5")

    take = move[:2]
    dest = move[2:]

    if take[0] not in "abcde" or dest[0] not in "abcde":
        raise ValueError("Columns must be a-e")

    if take[1] not in "12345" or dest[1] not in "12345":
        raise ValueError("Rows must be 1-5")

    return take, dest

def valid_take(board, r, c, player):
    if not is_border(r, c):
        raise ValueError("Must take from border")
    if board[r][c] not in (EMPTY, player):
        raise ValueError("Cannot take opponent piece")

def valid_insert(r, c, dest):
    dr, dc = coord_to_idx(dest)

    if (r, c) == (dr, dc):
        raise ValueError("Cannot reinsert at same position")

    # same row → must go to left or right edge
    if r == dr:
        if dc not in (0, SIZE - 1):
            raise ValueError("Row insert must be at aX or eX")
        return dr, dc

    # same column → must go to top or bottom edge
    if c == dc:
        if dr not in (0, SIZE - 1):
            raise ValueError("Column insert must be at X1 or X5")
        return dr, dc

    raise ValueError("Must stay in same row or column")

def push(board, r, c, dr, dc, player):
    if r == dr:
        if dc == 0:  # push from left
            for i in range(c, 0, -1):
                board[r][i] = board[r][i-1]
            board[r][0] = player
        elif dc == SIZE-1:  # push from right
            for i in range(c, SIZE-1):
                board[r][i] = board[r][i+1]
            board[r][SIZE-1] = player

    elif c == dc:
        if dr == 0:  # push from top
            for i in range(r, 0, -1):
                board[i][c] = board[i-1][c]
            board[0][c] = player
        elif dr == SIZE-1:  # push from bottom
            for i in range(r, SIZE-1):
                board[i][c] = board[i+1][c]
            board[SIZE-1][c] = player

def has_line(board, player):
    # rows and columns
    for i in range(SIZE):
        if all(board[i][j] == player for j in range(SIZE)):
            return True
        if all(board[j][i] == player for j in range(SIZE)):
            return True

    # diagonals
    if all(board[i][i] == player for i in range(SIZE)):
        return True
    if all(board[i][SIZE-1-i] == player for i in range(SIZE)):
        return True

    return False

def main():
    board = create_board()
    players = ["X", "O"]
    turn = 0
    move_count = 0
    MAX_MOVES = 200

    while True:
        player = players[turn % 2]
        opponent = players[(turn + 1) % 2]

        print_board(board)
        print(f"Player {player}")

        move = input("Enter move (e.g. a2a5): ").strip()

        try:
            take, dest = parse_move(move)
            r, c = coord_to_idx(take)

            valid_take(board, r, c, player)
            dr, dc = valid_insert(r, c, dest)

            push(board, r, c, dr, dc, player)

            player_line = has_line(board, player)
            opponent_line = has_line(board, opponent)

            # rule precedence: creating opponent line = immediate loss
            if opponent_line:
                print_board(board)
                print(f"{player} loses (formed {opponent}'s line)")
                break

            if player_line:
                print_board(board)
                print(f"{player} wins")
                break

            turn += 1
            move_count += 1
            if move_count >= MAX_MOVES:
                print_board(board)
                print("Draw — move limit reached.")
                break

        except ValueError as e:
            print("Error:", e)

if __name__ == "__main__":
    main()