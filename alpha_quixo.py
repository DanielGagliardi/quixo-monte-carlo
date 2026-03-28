import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from game_engine import Game, P1, P2, SIZE, EMPTY, BORDER_POSITIONS

# --- Constants & Move Mapping ---
# There are 44 possible moves in Quixo (16 border tiles, specific directions)
ALL_POSSIBLE_MOVES = []
for r, c in BORDER_POSITIONS:
    if c != 0:        ALL_POSSIBLE_MOVES.append((r, c, r, 0))
    if c != SIZE - 1: ALL_POSSIBLE_MOVES.append((r, c, r, SIZE - 1))
    if r != 0:        ALL_POSSIBLE_MOVES.append((r, c, 0, c))
    if r != SIZE - 1: ALL_POSSIBLE_MOVES.append((r, c, SIZE - 1, c))

MOVE_TO_IDX = {move: i for i, move in enumerate(ALL_POSSIBLE_MOVES)}
IDX_TO_MOVE = {i: move for i, move in enumerate(ALL_POSSIBLE_MOVES)}

# --- Neural Network Architecture ---

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
        return F.relu(x)

class QuixoNet(nn.Module):
    def __init__(self, num_res_blocks=4, num_hidden=64):
        super().__init__()
        # Input: 2 planes (Current Player, Opponent)
        self.start_block = nn.Sequential(
            nn.Conv2d(2, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * SIZE * SIZE, len(ALL_POSSIBLE_MOVES))
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * SIZE * SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_block(x)
        for res in self.res_blocks:
            x = res(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

# --- State Encoding ---

def get_encoded_state(board, player):
    """Encodes board into 2x5x5 binary planes relative to current player."""
    plane1 = np.zeros((SIZE, SIZE), dtype=np.float32)
    plane2 = np.zeros((SIZE, SIZE), dtype=np.float32)
    
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == player:
                plane1[r, c] = 1.0
            elif board[r][c] == -player: # Opponent
                plane2[r, c] = 1.0
                
    return torch.tensor(np.stack([plane1, plane2])).unsqueeze(0)

# --- MCTS with AlphaZero Logic ---

class MCTSNode:
    def __init__(self, game, state, player, parent=None, prior=0):
        self.game = game
        self.state = state
        self.player = player
        self.parent = parent
        
        self.children = {} # move_idx -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_move = -1
        best_child = None

        for move_idx, child in self.children.items():
            # PUCT Formula: Q + U
            u_score = c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            score = child.value + u_score
            
            if score > best_score:
                best_score = score
                best_move = move_idx
                best_child = child
        
        return best_move, best_child

    def expand(self, policy_logits, legal_moves):
        """Expands the node using the policy from the network."""
        # Masking: zero out illegal moves
        mask = np.zeros(len(ALL_POSSIBLE_MOVES), dtype=np.float32)
        legal_indices = [MOVE_TO_IDX[m] for m in legal_moves]
        mask[legal_indices] = 1.0
        
        # Apply mask and Softmax
        probs = F.softmax(policy_logits, dim=1).detach().cpu().numpy().flatten()
        probs *= mask
        if probs.sum() > 0:
            probs /= probs.sum() # Renormalize
        else:
            # Fallback if policy is dead: uniform over legal
            probs[legal_indices] = 1.0 / len(legal_indices)

        for m_idx in legal_indices:
            move = IDX_TO_MOVE[m_idx]
            next_state = self.game.next_state(self.state, move, self.player)
            self.children[m_idx] = MCTSNode(self.game, next_state, -self.player, parent=self, prior=probs[m_idx])

class AlphaZeroAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args # { 'c_puct': 1.4, 'num_simulations': 400, 'eps': 0.25, 'alpha': 0.3 }

    @torch.no_grad()
    def search(self, state, player):
        root = MCTSNode(self.game, state, player)
        
        # 1. Initial expansion of Root
        encoded = get_encoded_state(state, player).to(next(self.model.parameters()).device)
        policy_logits, _ = self.model(encoded)
        
        # --- Add Dirichlet Noise to Root ---
        legal = self.game.legal_moves(state, player)
        root.expand(policy_logits, legal)
        
        # Sample Dirichlet noise
        noise = np.random.dirichlet([self.args['alpha']] * len(root.children))
        for i, (m_idx, child) in enumerate(root.children.items()):
            child.prior = (1 - self.args['eps']) * child.prior + self.args['eps'] * noise[i]

        # 2. Run Simulations
        for _ in range(self.args['num_simulations']):
            node = root
            
            # Selection
            while node.children:
                _, node = node.select_child(self.args['c_puct'])
            
            # Check Terminal
            if self.game.is_terminal(node.state):
                # Note: reward() takes last_mover, so here last_mover was -node.player
                v = self.game.reward(node.state, -node.player)
            else:
                # Expansion & Evaluation
                encoded = get_encoded_state(node.state, node.player).to(next(self.model.parameters()).device)
                policy_logits, value = self.model(encoded)
                v = value.item()
                node.expand(policy_logits, self.game.legal_moves(node.state, node.player))
            
            # Backpropagation (v is from perspective of node.player)
            # We must flip v as we go up because nodes alternate players
            curr_v = v
            while node is not None:
                node.value_sum += curr_v
                node.visit_count += 1
                curr_v = -curr_v # Flip perspective
                node = node.parent
                
        # 3. Choose Action (Greedy for playing, proportional for training)
        best_move_idx = max(root.children, key=lambda i: root.children[i].visit_count)
        return IDX_TO_MOVE[best_move_idx]

# --- Main Play Loop Example ---
if __name__ == "__main__":
    game = Game()
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using device: {device}")

    model = QuixoNet().to(device)
    model.eval() # Set to eval for playing
    
    args = {
        'c_puct': 1.4,
        'num_simulations': 400,
        'eps': 0.25,
        'alpha': 0.3
    }
    
    agent = AlphaZeroAgent(model, game, args)
    
    state = game.initial_state()
    current_player = P1
    
    while not game.is_terminal(state):
        move = agent.search(state, current_player)
        state = game.next_state(state, move, current_player)
        print(f"Player {current_player} played {move}")
        current_player = -current_player
        
    print("Game Over. Winner Reward:", game.reward(state, -current_player))