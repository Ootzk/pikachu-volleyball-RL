"""라운드 로빈 ELO 평가 스크립트.

Usage:
  uv run python training/scripts/evaluate.py --players random,builtin,models/checkpoints/ppo_pikazoo --games 50
"""

import argparse

from training.utils.elo import make_player, round_robin


def main():
    parser = argparse.ArgumentParser(description="Round-robin ELO evaluation")
    parser.add_argument("--players", required=True,
                        help="Comma-separated player specs (random, builtin, or model path)")
    parser.add_argument("--games", type=int, default=100, help="Games per pair")
    parser.add_argument("--score", type=int, default=15, help="Winning score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    player_specs = [s.strip() for s in args.players.split(",")]
    players = [make_player(spec) for spec in player_specs]

    print(f"Players: {[p.name for p in players]}")
    print(f"Games per pair: {args.games}")
    print(f"Winning score: {args.score}\n")

    results, elos = round_robin(players, games_per_pair=args.games,
                                winning_score=args.score, seed=args.seed)

    # 대전 결과 출력
    print("=== Match Results ===")
    for (p1_name, p2_name), (wins, losses) in results.items():
        total = wins + losses
        win_pct = wins / total * 100
        print(f"{p1_name:20s} vs {p2_name:20s}: {wins:3d}W {losses:3d}L ({win_pct:5.1f}%)")

    # ELO 레이팅 출력
    print("\n=== ELO Ratings ===")
    sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    for name, elo in sorted_elos:
        print(f"{name:20s}: {elo:7.1f}")


if __name__ == "__main__":
    main()
