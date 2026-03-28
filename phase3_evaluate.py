"""
Full Phase 3 evaluation harness: UCT vs RAVE vs GRAVE.

Produces:
  1. Round-robin ELO table (all agent pairs, 50 games each)
  2. RAVE k ablation       (k ∈ {100, 300, 1000, 3000} vs UCT)
  3. GRAVE threshold ablation (t ∈ {5, 10, 50, 100, 200} vs UCT)
  4. Budget sensitivity    (win-rate vs sim count for UCT / RAVE / GRAVE)

"""

import math
import time
from game_engine import Game, P1, P2

from mcts_uct   import MCTSAgent, RandomAgent, GreedyAgent, play_game
from mcts_rave  import RAVEAgent
from mcts_grave import GRAVEAgent


# Generic game-pair evaluator 

def full_evaluate(game, agent1, agent2, n_games=50, silent=False):
    """
    Play n_games: half with agent1 as P1, half swapped.
    Returns {'W': wins, 'D': draws, 'L': losses} from agent1's perspective.
    """
    results = {'W': 0, 'D': 0, 'L': 0}
    half    = n_games // 2
    t0      = time.time()

    for _ in range(half):
        r = play_game(game, agent1, agent2)
        if r == 1: results['W'] += 1
        elif r == 0: results['D'] += 1
        else: results['L'] += 1

    for _ in range(half):
        r = -play_game(game, agent2, agent1)  # swap sides, negate for agent1
        if r == 1: results['W'] += 1
        elif r == 0: results['D'] += 1
        else: results['L'] += 1

    W, D, L = results['W'], results['D'], results['L']
    n       = W + D + L
    score   = (W + 0.5 * D) / n * 100

    if not silent:
        elapsed = time.time() - t0
        print(f"  {agent1.name:32s} vs {agent2.name:32s} | "
              f"W={W:3d} D={D:3d} L={L:3d} | score={score:5.1f}%  ({elapsed:.0f}s)")

    return results


# ELO computation 

def _elo_expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def compute_elo(agent_names, pair_results, base=1500, iterations=200, K=16):
    """
    Iterative ELO from all-pairs results dict.
    pair_results[(a, b)] = {'W': w, 'D': d, 'L': l}  (a's perspective)
    """
    ratings = {name: float(base) for name in agent_names}

    for _ in range(iterations):
        deltas = {name: 0.0 for name in agent_names}
        for (a, b), res in pair_results.items():
            n = res['W'] + res['D'] + res['L']
            if n == 0 or b not in ratings:
                continue
            score_a = (res['W'] + 0.5 * res['D']) / n
            ea      = _elo_expected(ratings[a], ratings[b])
            delta   = K * (score_a - ea)
            deltas[a] += delta
            deltas[b] -= delta
        for name in agent_names:
            ratings[name] += deltas[name]

    return ratings


# Main evaluation suite 

if __name__ == "__main__":
    game = Game()
    SIM  = 500

    # Define agent roster 
    agents = [
        RandomAgent(),
        GreedyAgent(),
        MCTSAgent(n_simulations=SIM),
        RAVEAgent(n_simulations=SIM,  k=300),
        RAVEAgent(n_simulations=SIM,  k=1000),
        GRAVEAgent(n_simulations=SIM, k=300, threshold=10),
        GRAVEAgent(n_simulations=SIM, k=300, threshold=50),
        GRAVEAgent(n_simulations=SIM, k=300, threshold=200),
    ]
    names = [a.name for a in agents]

    # Round-robin 
    print("=" * 90)
    print("ROUND ROBIN  (50 games per pair, 500 sims each)")
    print("=" * 90)

    pair_results = {}
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a1, a2 = agents[i], agents[j]
            res = full_evaluate(game, a1, a2, n_games=50)
            pair_results[(a1.name, a2.name)] = res
            pair_results[(a2.name, a1.name)] = {
                'W': res['L'], 'D': res['D'], 'L': res['W']
            }

    # ELO table 
    print("\n" + "=" * 55)
    print("ELO RATINGS")
    print("=" * 55)
    ratings = compute_elo(names, pair_results)
    for name, elo in sorted(ratings.items(), key=lambda x: -x[1]):
        print(f"  {name:40s}  {elo:7.1f}")

    # RAVE k ablation 
    print("\n" + "=" * 90)
    print("RAVE k ABLATION  (40 games vs UCT-500)")
    print("=" * 90)
    uct_ref = MCTSAgent(n_simulations=SIM)
    for k_val in [100, 300, 1000, 3000]:
        full_evaluate(game, RAVEAgent(SIM, k=k_val), uct_ref, n_games=40)

    # GRAVE threshold ablation 
    print("\n" + "=" * 90)
    print("GRAVE THRESHOLD ABLATION  (40 games vs UCT-500)")
    print("=" * 90)
    for t in [5, 10, 50, 100, 200]:
        full_evaluate(game, GRAVEAgent(SIM, k=300, threshold=t), uct_ref, n_games=40)

    # Budget sensitivity 
    print("\n" + "=" * 70)
    print("BUDGET SENSITIVITY vs Random (20 games per point)")
    print(f"{'Sims':>6}  {'UCT':>8}  {'RAVE-300':>10}  {'GRAVE-t50':>11}")
    print("-" * 42)
    ref = RandomAgent()
    for sims in [50, 100, 200, 500, 1000, 2000]:
        def score(agent):
            r = full_evaluate(game, agent, ref, n_games=20, silent=True)
            n = r['W'] + r['D'] + r['L']
            return (r['W'] + 0.5 * r['D']) / n * 100

        uct_s   = score(MCTSAgent(sims))
        rave_s  = score(RAVEAgent(sims,  k=300))
        grave_s = score(GRAVEAgent(sims, k=300, threshold=50))
        print(f"{sims:>6}  {uct_s:>7.1f}%  {rave_s:>9.1f}%  {grave_s:>10.1f}%")

    # Draw-rate analysis (RAVE vs GRAVE at high budget) 
    print("\n" + "=" * 90)
    print("DRAW RATE: GRAVE-50 vs RAVE-300  (100 games, 1000 sims)")
    print("(Stronger agents → higher mutual draw rate → approaching optimal play)")
    print("=" * 90)
    full_evaluate(game,
                  GRAVEAgent(1000, k=300, threshold=50),
                  RAVEAgent(1000,  k=300),
                  n_games=100)
