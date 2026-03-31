import pygame
import sys

SIZE = 5
CELL = 100
MARGIN = 50
EMPTY = ""

WIDTH = HEIGHT = SIZE * CELL + 2 * MARGIN

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Quixo")

font = pygame.font.SysFont(None, 48)

class Game:
    def __init__(self):
        self.board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
        self.player = "X"
        self.selected = None
        self.drag_pos = None
        self.running = True

    def draw(self):
        screen.fill((30, 30, 30))

        for r in range(SIZE):
            for c in range(SIZE):
                x = MARGIN + c * CELL
                y = MARGIN + r * CELL

                pygame.draw.rect(screen, (200, 200, 200), (x, y, CELL, CELL), 2)

                val = self.board[r][c]
                if val:
                    txt = font.render(val, True, (255, 255, 255))
                    screen.blit(txt, (x + CELL//2 - 10, y + CELL//2 - 20))

        # draw dragged piece
        if self.selected and self.drag_pos:
            r, c = self.selected
            dst = self.get_cell(self.drag_pos)
            
            # Check if destination is valid
            if dst and self.valid_insert(r, c, dst[0], dst[1]):
                color = (0, 255, 0)  # green for valid
            else:
                color = (255, 100, 100)  # red for invalid
            
            txt = font.render(self.player, True, color)
            screen.blit(txt, (self.drag_pos[0] - 15, self.drag_pos[1] - 20))

        pygame.display.flip()

    def get_cell(self, pos):
        x, y = pos
        c = (x - MARGIN) // CELL
        r = (y - MARGIN) // CELL
        if 0 <= r < SIZE and 0 <= c < SIZE:
            return r, c
        return None

    def is_border(self, r, c):
        return r in (0, SIZE-1) or c in (0, SIZE-1)

    def valid_take(self, r, c):
        return self.is_border(r, c) and self.board[r][c] in (EMPTY, self.player)

    def valid_insert(self, r, c, dr, dc):
        if (r, c) == (dr, dc):
            return False
        if r == dr and dc in (0, SIZE-1):
            return True
        if c == dc and dr in (0, SIZE-1):
            return True
        return False

    def push(self, r, c, dr, dc):
        p = self.player

        if r == dr:
            if dc == 0:
                for i in range(c, 0, -1):
                    self.board[r][i] = self.board[r][i-1]
                self.board[r][0] = p
            else:
                for i in range(c, SIZE-1):
                    self.board[r][i] = self.board[r][i+1]
                self.board[r][SIZE-1] = p

        elif c == dc:
            if dr == 0:
                for i in range(r, 0, -1):
                    self.board[i][c] = self.board[i-1][c]
                self.board[0][c] = p
            else:
                for i in range(r, SIZE-1):
                    self.board[i][c] = self.board[i+1][c]
                self.board[SIZE-1][c] = p

    def has_line(self, player):
        for i in range(SIZE):
            if all(self.board[i][j] == player for j in range(SIZE)):
                return True
            if all(self.board[j][i] == player for j in range(SIZE)):
                return True

        if all(self.board[i][i] == player for i in range(SIZE)):
            return True
        if all(self.board[i][SIZE-1-i] == player for i in range(SIZE)):
            return True

        return False

    def end(self, text):
        print(text)
        pygame.quit()
        sys.exit()

game = Game()

while game.running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            cell = game.get_cell(event.pos)
            if cell:
                r, c = cell
                if game.valid_take(r, c):
                    game.selected = (r, c)
                    game.drag_pos = event.pos

        elif event.type == pygame.MOUSEMOTION:
            if game.selected:
                game.drag_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if game.selected:
                src = game.selected
                dst = game.get_cell(event.pos)

                if dst:
                    r, c = src
                    dr, dc = dst

                    if game.valid_insert(r, c, dr, dc):
                        game.push(r, c, dr, dc)

                        player = game.player
                        opponent = "O" if player == "X" else "X"

                        if game.has_line(opponent):
                            game.draw()
                            game.end(f"{player} loses")

                        if game.has_line(player):
                            game.draw()
                            game.end(f"{player} wins")

                        game.player = opponent

                game.selected = None
                game.drag_pos = None

    game.draw()