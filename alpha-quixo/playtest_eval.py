import numpy as np
print(np.__version__)


import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

from tqdm.notebook import trange

import random
import math
import time

SIZE = 5
EMPTY = 0
P1 = 1
P2 = -1
max_moves = 100


class Quixo:
    def __init__(self):
        self.size = SIZE

        # Precompute perimeter positions
        self.perimeter = [
            (r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if r == 0 or r == SIZE - 1 or c == 0 or c == SIZE - 1
        ]

        # Max 3 moves per position → upper bound
        self.max_moves = len(self.perimeter) * 3
        self.action_map = []  # (r, c, dr, dc)
        self._build_action_map()

        self.action_size = len(self.action_map)

    def __repr__(self):
        return "Quixo"

    # ---------- State ----------

    def get_initial_state(self):
        return np.zeros((self.size, self.size), dtype=np.int8)

    def change_perspective(self, state, player):
        return state * player

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    # ---------- Encoding ----------

    def get_encoded_state(self, state):
        encoded = np.stack(
            (state == P2, state == EMPTY, state == P1)
        ).astype(np.float32)

        if state.ndim == 3:
            encoded = np.swapaxes(encoded, 0, 1)

        return encoded

    # ---------- Action Space ----------

    def _build_action_map(self):
        """Enumerate all *legal* (position, direction) pairs."""
        for (r, c) in self.perimeter:

            # possible insertions excluding same-side
            candidates = []

            if c != 0:
                candidates.append((r, c, r, 0))
            if c != self.size - 1:
                candidates.append((r, c, r, self.size - 1))
            if r != 0:
                candidates.append((r, c, 0, c))
            if r != self.size - 1:
                candidates.append((r, c, self.size - 1, c))

            self.action_map.extend(candidates)

    def encode_action(self, r, c, dr, dc):
        return self.action_map.index((r, c, dr, dc))

    def decode_action(self, action):
        return self.action_map[action]

    # ---------- Legal Moves ----------

    def get_valid_moves(self, state):
        """
        Assumes state is already in current player's perspective (player = +1).
        """
        valid = np.zeros(self.action_size, dtype=np.uint8)

        for i, (r, c, dr, dc) in enumerate(self.action_map):
            piece = state[r, c]

            # can take EMPTY or own piece only
            if piece == EMPTY or piece == P1:
                valid[i] = 1

        return valid

    # ---------- Game Mechanics ----------

    def push(self, state, move):
        """
        Apply move assuming current player is +1 (canonical form).
        """
        r, c, dr, dc = move
        new = state.copy()

        # remove cube
        if r == dr:  # row move
            row = list(new[r, :])
            row.pop(c)

            # insert with forced overwrite
            if dc == 0:
                row.insert(0, P1)
            else:
                row.append(P1)

            new[r, :] = row

        else:  # column move
            col = list(new[:, c])
            col.pop(r)

            if dr == 0:
                col.insert(0, P1)
            else:
                col.append(P1)

            new[:, c] = col

        return new

    def get_next_state(self, state, action, player):
        """
        Applies action in *absolute* space, not canonical.
        """
        move = self.decode_action(action)

        # convert to canonical
        canonical = self.change_perspective(state, player)

        next_state = self.push(canonical, move)

        # revert perspective
        return self.change_perspective(next_state, player)

    # ---------- Win Detection ----------

    def has_line(self, state, player):
        # rows
        for r in range(self.size):
            if np.all(state[r, :] == player):
                return True

        # columns
        for c in range(self.size):
            if np.all(state[:, c] == player):
                return True

        # diagonals
        if np.all(np.diag(state) == player):
            return True

        if np.all(np.diag(np.fliplr(state)) == player):
            return True

        return False

    # ---------- Value ----------

    def get_value_and_terminated(self, state, player, move_count=None):
        """
        Evaluate from current player's perspective.

        If move_count is provided and >= max_moves with no winner, declare draw.
        """

        player_win = self.has_line(state, player)
        opp_win = self.has_line(state, -player)

        # critical rule: opponent line dominates
        if opp_win:
            return -1, True

        if player_win:
            return 1, True

        if move_count is not None and move_count >= max_moves:
            return 0, True

        return 0, False
    
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.size * game.size, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.size * game.size, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
        
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0, move_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.move_count = move_count
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(
                    self.game,
                    self.args,
                    child_state,
                    self,
                    action,
                    prob,
                    move_count=self.move_count + 1,
                )
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1, move_count=0)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, 1, move_count=node.move_count)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        

def print_board(state):
    symbols = {0: ".", 1: "X", -1: "O"}
    print()
    for r in range(state.shape[0]):
        print(" ".join(symbols[x] for x in state[r]))
    print()


def get_human_action(game, state):
    side_map = {
        "L": lambda r, c: (r, c, r, 0),
        "R": lambda r, c: (r, c, r, game.size - 1),
        "T": lambda r, c: (r, c, 0, c),
        "B": lambda r, c: (r, c, game.size - 1, c),
    }

    valid = game.get_valid_moves(state)

    while True:
        try:
            raw = input("Enter move (r c side[L/R/T/B]): ").strip().split()
            r, c = int(raw[0]), int(raw[1])
            side = raw[2].upper()

            if side not in side_map:
                raise ValueError

            move = side_map[side](r, c)

            if move not in game.action_map:
                print("Invalid geometry")
                continue

            action = game.encode_action(*move)

            if valid[action] == 0:
                print("Illegal move")
                continue

            return action

        except Exception:
            print("Invalid input format")


# --- Setup ---
game = Quixo()
player = 1

args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("model_fast_2026_03_30_17_04_06_4_<__main__.QuixoFast object at 0x31b604b90>.pt", map_location=device))
model.eval()

mcts = MCTS(game, args, model)

state = game.get_initial_state()

import random

# --- Random policy ---
def get_random_action(game, state):
    valid = game.get_valid_moves(state)
    valid_indices = np.where(valid == 1)[0]
    return int(np.random.choice(valid_indices))

from tqdm import tqdm
import numpy as np

from tqdm import tqdm
import numpy as np

def evaluate(model, game, mcts, num_games=50):
    wins = 0
    losses = 0
    draws = 0

    pbar = tqdm(range(num_games))

    for i in pbar:
        state = game.get_initial_state()
        player = 1 if i % 2 == 0 else -1
        move_count = 0

        while True:
            if player == 1:
                neutral_state = game.change_perspective(state, player)
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)
            else:
                neutral_state = game.change_perspective(state, player)
                action = get_random_action(game, neutral_state)

            state = game.get_next_state(state, action, player)
            move_count += 1

            value, is_terminal = game.get_value_and_terminated(state, player, move_count)

            if is_terminal:
                if value == 1:
                    if player == 1:
                        wins += 1
                    else:
                        losses += 1
                elif value == -1:
                    if player == 1:
                        losses += 1
                    else:
                        wins += 1
                else:
                    draws += 1
                break

            player = game.get_opponent(player)

        pbar.set_postfix({
            "W": wins,
            "L": losses,
            "D": draws
        })

    total = num_games
    print("Results over", total, "games")
    print("Win %:", wins / total * 100)
    print("Loss %:", losses / total * 100)
    print("Draw %:", draws / total * 100)


# Run
evaluate(model, game, mcts, num_games=50)