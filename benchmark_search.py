# benchmark_search.py
"""
Systematic hyperparameter search for the best MCTS / RAVE / GRAVE agents.

Based on prior results:
  - GRAVE(500, k=300, t=5)  →  66% vs UCT-500  (p ≈ 0.012, significant)
  - RAVE( 500, k=3000)      →  62% vs UCT-500  (p ≈ 0.045, borderline)
  - Budget: MCTS(1000)       →  72.5% vs MCTS(500)
  - C=0 (no exploration) used throughout — untested whether C>0 helps

Search plan
───────────
  Stage 1 — Confirm top candidates        (100 games, high confidence)
  Stage 2 — GRAVE fine threshold search   (50 games, t ∈ {1,2,3,8,15,25})
  Stage 3 — GRAVE k × t joint grid        (50 games, k ∈ {1k,3k,5k} × t ∈ {1,5})
  Stage 4 — RAVE high-k exploration       (50 games, k ∈ {5k,10k,20k})
  Stage 5 — UCT / RAVE / GRAVE C ablation (50 games, C ∈ {0.1,0.3,0.5,1.0,√2})
  Stage 6 — Budget scaling                (50 games, best variants at 1000 sims)
  Stage 7 — Championship round-robin      (100 games, auto-selected top agents)

Resume behavior
───────────────
  Re-running this script is safe and idempotent: every finished matchup is
  loaded from its checkpoint instantly without replaying any games. Only
  unfinished or missing matchups are actually computed.
  Force a full re-run by passing  resume=False  to individual evaluate() calls,
  or by deleting the relevant files in  checkpoints/ .
"""

import math
import random
import time
import json
import os
import logging
import itertools
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from game_engine import Game, P1, P2
from mcts_uct   import MCTSAgent, RandomAgent, GreedyAgent
from mcts_rave  import RAVEAgent
from mcts_grave import GRAVEAgent

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_GAME_MOVES  = 400
LOGS_DIR        = "logs"
CHECKPOINTS_DIR = "checkpoints"

os.makedirs(LOGS_DIR,        exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ── Logging / Checkpointing ────────────────────────────────────────────────────

def _checkpoint_path(run_id: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, f"{run_id}.json")

def _log_path(run_id: str) -> str:
    return os.path.join(LOGS_DIR, f"{run_id}.log")

def _setup_logger(run_id: str, mode: str = "a") -> logging.Logger:
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh  = logging.FileHandler(_log_path(run_id), mode=mode, encoding="utf-8")
    fh.setFormatter(fmt)
    ch  = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

def _save_checkpoint(run_id: str, data: dict) -> None:
    path, tmp = _checkpoint_path(run_id), _checkpoint_path(run_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _load_checkpoint(run_id: str) -> Optional[dict]:
    path = _checkpoint_path(run_id)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None

# ── Agent factory (ensures unique, descriptive names + registry) ───────────────

AGENT_REGISTRY: Dict[str, object] = {}

def _reg(ag):
    AGENT_REGISTRY[ag.name] = ag
    return ag

def mcts(n: int, c: float = 0.0):
    ag      = MCTSAgent(n, c=c)
    ag.name = f"MCTS({n})" if c == 0.0 else f"MCTS({n},C={c:.2g})"
    return _reg(ag)

def rave(n: int, k: int, c: float = 0.0):
    ag      = RAVEAgent(n, k=k, c=c)
    ag.name = f"RAVE({n},k={k})" if c == 0.0 else f"RAVE({n},k={k},C={c:.2g})"
    return _reg(ag)

def grave(n: int, k: int, t: int, c: float = 0.0):
    ag      = GRAVEAgent(n, k=k, threshold=t, c=c)
    ag.name = (f"GRAVE({n},k={k},t={t})" if c == 0.0
               else f"GRAVE({n},k={k},t={t},C={c:.2g})")
    return _reg(ag)

# ── Game loop ──────────────────────────────────────────────────────────────────

def play_game(game, agent1, agent2, verbose: bool = False) -> int:
    """Returns +1 agent1 wins, -1 agent2 wins, 0 draw."""
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
        return 0
    last_mover = -player
    return game.reward(state, last_mover) * last_mover

# ── Evaluate (resume-aware, idempotent) ────────────────────────────────────────

def evaluate(
    game,
    agent1,
    agent2,
    n_games: int      = 100,
    verbose: bool     = False,
    run_id: Optional[str] = None,
    resume: bool      = True,
) -> dict:
    """
    Run n_games between agent1 and agent2 (half each side).
    Logs every game to  logs/<run_id>.log  and checkpoints after each game.

    Idempotency: if a finished checkpoint for this run_id already exists,
    the cached result is returned immediately without playing any new games.
    Pass  resume=False  to force a fresh run (new timestamped run_id).
    """
    base_id    = run_id or f"{agent1.name}_vs_{agent2.name}_{n_games}g"
    checkpoint = _load_checkpoint(base_id) if resume else None

    # ── Already finished → return cache ──────────────────────────────────────
    if checkpoint and checkpoint.get("finished"):
        logger = _setup_logger(base_id)
        logger.info(f"'{base_id}' already complete — using cached result.")
        return {"W": checkpoint["W"], "D": checkpoint["D"], "L": checkpoint["L"]}

    # ── Fresh or partial run ──────────────────────────────────────────────────
    actual_id = base_id
    if not resume:
        actual_id  = f"{base_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint = None

    logger = _setup_logger(actual_id)

    if checkpoint:
        outcomes, start_game = checkpoint["outcomes"], checkpoint["completed_games"]
        logger.info(f"Resuming '{actual_id}' — {start_game}/{n_games} done.")
    else:
        outcomes, start_game = [], 0
        logger.info(f"Starting '{actual_id}' — {n_games} games "
                    f"({agent1.name} vs {agent2.name})")

    half = n_games // 2
    t0   = time.time()

    def _run_and_save(idx: int, a1, a2, label: str) -> None:
        r      = play_game(game, a1, a2, verbose)
        result = r if label == "a1=P1" else -r
        outcomes.append(result)
        sym    = {1: "W", 0: "D", -1: "L"}[result]
        logger.info(f"  [{sym}] game {idx+1:3d}/{n_games}  ({label})  "
                    f"raw={r:+d}  a1={result:+d}")
        _save_checkpoint(actual_id, {
            "run_id": actual_id, "agent1": agent1.name, "agent2": agent2.name,
            "n_games": n_games, "completed_games": len(outcomes),
            "outcomes": outcomes, "finished": False,
        })

    for i in range(half):
        if i < start_game: continue
        _run_and_save(i, agent1, agent2, "a1=P1")
        print(f"  game {i+1:3d}/{half} (a1=P1)", end="\r")

    for i in range(half):
        g = half + i
        if g < start_game: continue
        _run_and_save(g, agent2, agent1, "a1=P2")
        print(f"  game {i+1:3d}/{half} (a1=P2)", end="\r")

    W, D, L = outcomes.count(1), outcomes.count(0), outcomes.count(-1)
    n_done  = len(outcomes)
    elapsed = time.time() - t0

    summary = (
        f"\n{'─'*54}\n"
        f"  {agent1.name:<26}vs {agent2.name}\n"
        f"  Games  : {n_done}/{n_games}  ({elapsed:.1f}s)\n"
        f"  Wins   : {W:3d} ({100*W/n_done:5.1f}%)\n"
        f"  Draws  : {D:3d} ({100*D/n_done:5.1f}%)\n"
        f"  Losses : {L:3d} ({100*L/n_done:5.1f}%)\n"
        f"{'─'*54}"
    )
    logger.info(summary)

    _save_checkpoint(actual_id, {
        "run_id": actual_id, "agent1": agent1.name, "agent2": agent2.name,
        "n_games": n_games, "completed_games": n_done,
        "outcomes": outcomes, "finished": True,
        "W": W, "D": D, "L": L,
    })
    return {"W": W, "D": D, "L": L}

# ── Leaderboard ────────────────────────────────────────────────────────────────

class Leaderboard:
    """
    Aggregates evaluate() results.  For each agent tracks:
      • overall W/D/L across all games played
      • W/D/L specifically against the reference agent
    Prints a sorted table with win-rates and a significance indicator.
      ✓  |z| ≥ 1.96  →  p < 0.05
      ~  |z| ≥ 1.28  →  p < 0.10
         not significant
    """

    _SIG   = 1.96
    _TREND = 1.28

    def __init__(self, reference_name: str = "MCTS(500)"):
        self.reference_name = reference_name
        self._records: List[dict] = []

    def add(self, results: dict, agent1_name: str, agent2_name: str) -> None:
        self._records.append({
            "agent1": agent1_name, "agent2": agent2_name,
            "W": results["W"], "D": results["D"], "L": results["L"],
        })

    def _agent_stats(self) -> Dict[str, dict]:
        stats: Dict[str, dict] = {}
        def _e(name):
            if name not in stats:
                stats[name] = {"W":0,"D":0,"L":0,"rW":0,"rD":0,"rL":0}
        for r in self._records:
            a1, a2 = r["agent1"], r["agent2"]
            _e(a1); _e(a2)
            stats[a1]["W"] += r["W"]; stats[a1]["D"] += r["D"]; stats[a1]["L"] += r["L"]
            stats[a2]["W"] += r["L"]; stats[a2]["D"] += r["D"]; stats[a2]["L"] += r["W"]
            if a2 == self.reference_name:
                stats[a1]["rW"] += r["W"]; stats[a1]["rD"] += r["D"]
                stats[a1]["rL"] += r["L"]
            if a1 == self.reference_name:
                stats[a2]["rW"] += r["L"]; stats[a2]["rD"] += r["D"]
                stats[a2]["rL"] += r["W"]
        return stats

    @staticmethod
    def _z(w, n):
        return (w / n - 0.5) / math.sqrt(0.25 / n) if n > 0 else 0.0

    @staticmethod
    def _wr(w, d, l):
        n = w + d + l
        return 100 * w / n if n > 0 else float("nan")

    def print_standings(self, logger: Optional[logging.Logger] = None) -> None:
        stats = self._agent_stats()
        rows  = []
        for name, s in stats.items():
            an   = s["W"]  + s["D"]  + s["L"]
            rn   = s["rW"] + s["rD"] + s["rL"]
            awr  = self._wr(s["W"],  s["D"],  s["L"])
            rwr  = self._wr(s["rW"], s["rD"], s["rL"])
            z    = self._z(s["rW"], rn) if rn > 0 else 0.0
            sig  = "✓" if abs(z) >= self._SIG else "~" if abs(z) >= self._TREND else " "
            key  = rwr if rn >= 20 else awr
            rows.append((name, awr, an, rwr, rn, sig, key))

        rows.sort(key=lambda r: -r[6])
        ref    = self.reference_name
        header = (
            f"\n{'═'*74}\n"
            f"  {'AGENT':<34} {'vs ALL':>7} {'N':>5}  "
            f"{'vs ' + ref:>12} {'N':>5}  Sig\n"
            f"{'─'*74}"
        )
        lines  = [header]
        for name, awr, an, rwr, rn, sig, _ in rows:
            rwr_s = f"{rwr:6.1f}%" if rn > 0 else "    n/a"
            lines.append(
                f"  {name:<34} {awr:6.1f}% {an:5d}  {rwr_s:>12} {rn:5d}   {sig}"
            )
        lines.append("═" * 74)
        text = "\n".join(lines)
        print(text)
        if logger:
            logger.info(text)

# ── Stage runner ───────────────────────────────────────────────────────────────

def run_stage(
    game,
    stage_name: str,
    matchups: List[Tuple],
    lb: Leaderboard,
    logger: logging.Logger,
) -> None:
    sep = "#" * 56
    logger.info(f"\n{sep}\n  {stage_name}\n{sep}")
    for agent1, agent2, n_games in matchups:
        try:
            results = evaluate(game, agent1, agent2, n_games=n_games)
            lb.add(results, agent1.name, agent2.name)
        except KeyboardInterrupt:
            logger.warning("Interrupted — saving partial leaderboard.")
            lb.print_standings(logger)
            raise
    lb.print_standings(logger)

# ── Champion auto-selection ────────────────────────────────────────────────────

def select_champions(
    lb: Leaderboard,
    ref_name: str  = "MCTS(500)",
    min_games: int = 40,
    top_n: int     = 6,
) -> List[str]:
    """
    Pick top_n distinct agents ranked by win-rate vs ref_name.
    Requires at least min_games played vs the reference.
    The reference agent itself is always included.
    """
    stats      = lb._agent_stats()
    candidates = []
    for name, s in stats.items():
        rn = s["rW"] + s["rD"] + s["rL"]
        if rn >= min_games and name != ref_name:
            candidates.append((name, 100 * s["rW"] / rn))
    candidates.sort(key=lambda x: -x[1])
    top = [name for name, _ in candidates[:top_n - 1]]
    top.append(ref_name)
    return top

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    game    = Game()
    lb      = Leaderboard(reference_name="MCTS(500)")
    _sqrt2  = math.sqrt(2)
    _ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger  = _setup_logger(f"benchmark_{_ts}", mode="w")

    logger.info("=" * 56)
    logger.info("  MCTS / RAVE / GRAVE Hyperparameter Search")
    logger.info(f"  Started: {_ts}")
    logger.info("=" * 56)

    # ── Reference pool ────────────────────────────────────────────────────────
    REF   = mcts(500)    # "MCTS(500)"  — main reference
    REF1K = mcts(1000)   # "MCTS(1000)" — budget reference

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Confirm top candidates  (100 games each)
    #
    #  Goal: high-confidence estimates for the two standout variants from
    #        prior results and their direct head-to-head.
    # ─────────────────────────────────────────────────────────────────────────
    run_stage(game, "STAGE 1 — Confirm top candidates", [
        (grave(500, k=300,  t=5),    REF,                   100),
        (rave( 500, k=3000),         REF,                   100),
        (grave(500, k=300,  t=5),    rave(500, k=3000),     100),
    ], lb, logger)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2 — GRAVE fine threshold search  (50 games each)
    #
    #  Goal: explore t < 5 (possibly better) and fill the gap t ∈ {8,15,25}
    #        to map the win-rate curve precisely around the sweet spot.
    # ─────────────────────────────────────────────────────────────────────────
    run_stage(game, "STAGE 2 — GRAVE fine threshold search", [
        (grave(500, k=300, t=t), REF, 50)
        for t in [1, 2, 3, 8, 15, 25]
    ], lb, logger)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3 — GRAVE k × t joint grid  (50 games each)
    #
    #  Goal: does GRAVE also benefit from higher k (as RAVE did at k=3000)?
    # ─────────────────────────────────────────────────────────────────────────
    run_stage(game, "STAGE 3 — GRAVE k × t joint grid", [
        (grave(500, k=k, t=t), REF, 50)
        for k in [1000, 3000, 5000]
        for t in [1, 5]
    ], lb, logger)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4 — RAVE high-k exploration  (50 games each)
    #
    #  Goal: k=3000 was promising — does the trend continue further?
    # ─────────────────────────────────────────────────────────────────────────
    run_stage(game, "STAGE 4 — RAVE high-k exploration", [
        (rave(500, k=k), REF, 50)
        for k in [5000, 10000, 20000]
    ], lb, logger)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5 — Exploration constant C ablation  (50 games each)
    #
    #  Goal: all prior runs used C=0 (pure exploitation). Does adding UCT
    #        exploration help UCT, RAVE, or GRAVE?
    # ─────────────────────────────────────────────────────────────────────────
    run_stage(game, "STAGE 5 — Exploration constant C ablation", [
        (mcts(500, c=c),              REF, 50) for c in [0.1, 0.3, 0.5, 1.0, _sqrt2]
    ] + [
        (grave(500, k=300, t=5, c=c), REF, 50) for c in [0.1, 0.5, 1.0]
    ] + [
        (rave(500, k=3000, c=c),      REF, 50) for c in [0.1, 0.5, 1.0]
    ], lb, logger)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6 — Budget scaling  (50 games each)
    #
    #  Goal: does the best algorithm variant at 1000 sims beat MCTS(1000)?
    # ─────────────────────────────────────────────────────────────────────────
    run_stage(game, "STAGE 6 — Budget scaling", [
        (grave(1000, k=300,  t=5),  REF1K, 50),
        (rave( 1000, k=3000),       REF1K, 50),
        (grave(1000, k=3000, t=5),  REF1K, 50),
        (grave(1000, k=300,  t=5),  REF,   50),   # also vs main ref
        (rave( 1000, k=3000),       REF,   50),
    ], lb, logger)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 7 — Championship round-robin  (100 games per pair)
    #
    #  Auto-selects top 6 agents by confirmed win-rate vs MCTS(500),
    #  then runs every unique pair for the definitive head-to-head ranking.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n── Auto-selecting champions from Stages 1–6 ──")
    champ_names  = select_champions(lb, ref_name="MCTS(500)", min_games=40, top_n=6)
    logger.info(f"  Selected: {champ_names}")

    champ_agents = [AGENT_REGISTRY[n] for n in champ_names if n in AGENT_REGISTRY]
    pairs        = list(itertools.combinations(champ_agents, 2))
    logger.info(f"  → {len(champ_agents)} agents, {len(pairs)} pairs\n")

    run_stage(game,
              f"STAGE 7 — Championship ({len(champ_agents)} agents, {len(pairs)} pairs)",
              [(a1, a2, 100) for a1, a2 in pairs],
              lb, logger)

    # ── Final leaderboard ─────────────────────────────────────────────────────
    logger.info("\n\n" + "═" * 74)
    logger.info("  FINAL LEADERBOARD  —  all stages combined")
    logger.info("═" * 74)
    lb.print_standings(logger)
    logger.info(f"\nDone.  Full log: {_log_path('benchmark_' + _ts)}")