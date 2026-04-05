import math
import json
import os
import time
from game_engine import Game
from mcts_grave import GRAVEAgent, evaluate

def analyze_results(run_id, total_games):
    """Parses the JSON checkpoint to generate the 3-layer final report."""
    checkpoint_path = os.path.join("checkpoints", f"{run_id}.json")
    if not os.path.exists(checkpoint_path):
        print(f"Could not find checkpoint {checkpoint_path} for analysis.")
        return

    with open(checkpoint_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    outcomes = data.get("outcomes", [])
    if len(outcomes) != total_games:
        print(f"Warning: Expected {total_games} games, but found {len(outcomes)}.")
    
    half = total_games // 2
    
    # outcomes array: 
    # First half -> Agent 1 is P1. (1 = P1 Win, -1 = P2 Win)
    # Second half -> Agent 1 is P2. (1 = P2 Win, -1 = P1 Win)
    p1_wins = outcomes[:half].count(1) + outcomes[half:].count(-1)
    p2_wins = outcomes[:half].count(-1) + outcomes[half:].count(1)
    draws = outcomes.count(0)
    
    print("\n" + "="*60)
    print("FINAL REPORT: GRAVE(1000) Self-Play Analysis")
    print("="*60)
    
    # 1. Raw Counts and Percentages
    print(f"Total Games : {total_games}")
    print(f"P1 Wins     : {p1_wins} ({(p1_wins/total_games)*100:.1f}%)")
    print(f"P2 Wins     : {p2_wins} ({(p2_wins/total_games)*100:.1f}%)")
    print(f"Draws       : {draws} ({(draws/total_games)*100:.1f}%)")
    
    # 2. Z-score Test (vs 50% null hypothesis on decisive games)
    total_decisive = p1_wins + p2_wins
    if total_decisive > 0:
        p1_win_rate = p1_wins / total_decisive
        # Z = (p - p0) / sqrt(p0 * q0 / n)
        z_score = (p1_win_rate - 0.5) / math.sqrt(0.25 / total_decisive)
        print(f"\nZ-Score (P1 vs P2 Decisive): {z_score:.2f}")
    else:
        print("\nZ-Score (P1 vs P2 Decisive): N/A (All games drawn)")

    # 3. Symmetry Check
    divergence = abs((p1_wins / total_games) - (p2_wins / total_games)) * 100
    print(f"\nP1 vs P2 Win Rate Divergence : {divergence:.1f}%")
    if divergence > 10.0:
        print("  -> Warning: >10% divergence indicates a potential first-mover advantage at this budget.")
    else:
        print("  -> Divergence is within acceptable bounds (<= 10%).")
        
    # Interpretation Table Logic
    print("\nDRAW RATE INTERPRETATION:")
    draw_rate = (draws / total_games) * 100
    if draw_rate >= 80:
        print(f"[✓] {draw_rate:.1f}% : Strong empirical support for the theoretical draw result.")
    elif draw_rate >= 60:
        print(f"[~] {draw_rate:.1f}% : Moderate — agent approximates draw strategy but budget is limiting.")
    elif draw_rate >= 40:
        print(f"[?] {draw_rate:.1f}% : Weak — try GRAVE(2000) or higher.")
    else:
        print(f"[✗] {draw_rate:.1f}% : Inconsistent — likely misconfiguration.")
    print("="*60)

def main():
    print("Initializing Quixo Game Engine...")
    game = Game()

    # Design Decision: Two separate agent instances with C=0.0
    print("Loading GRAVE Agents...")
    agent1 = GRAVEAgent(n_simulations=1000, k=3000, threshold=5, c=0.0)
    agent2 = GRAVEAgent(n_simulations=1000, k=3000, threshold=5, c=0.0)

    agent1.name = "GRAVE_P1(1000,k3000,t5,c0)"
    agent2.name = "GRAVE_P2(1000,k3000,t5,c0)"

    total_games = 100
    run_id = "GRAVE_SelfPlay_1000_k3000_t5_c0"

    print(f"Starting Self-Play: {agent1.name} vs {agent2.name} for {total_games} games.")
    print(f"Logs and checkpoints will be saved under run_id: {run_id}")
    print("-" * 60)

    start_time = time.time()
    
    # evaluate() handles the atomic os.replace() checkpointing and side swapping
    evaluate(
        game=game,
        agent1=agent1,
        agent2=agent2,
        n_games=total_games,
        verbose=False,
        run_id=run_id,
        resume=True
    )

    elapsed = time.time() - start_time
    print(f"\nSelf-Play Execution Complete in {elapsed:.2f} seconds.")
    
    # Execute the custom analysis directly from the crash-safe checkpoint
    analyze_results(run_id, total_games)

if __name__ == "__main__":
    main()