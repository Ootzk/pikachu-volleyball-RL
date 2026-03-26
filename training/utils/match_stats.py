"""게임 상세 통계 추출."""

from dataclasses import dataclass, field

import numpy as np

from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import NormalizeObservation, SimplifyAction
from training.utils.elo import Player, make_player


@dataclass
class RoundStats:
    server: str  # "player_1" or "player_2"
    winner: str  # "player_1" or "player_2"
    rally_length: int  # 해당 라운드의 스텝 수


@dataclass
class GameStats:
    p1_score: int = 0
    p2_score: int = 0
    rounds: list = field(default_factory=list)
    truncated_rallies: int = 0
    seed: int = None

    @property
    def winner(self):
        if self.p1_score == self.p2_score:
            return "draw"
        return "player_1" if self.p1_score > self.p2_score else "player_2"

    @property
    def p1_serve_win(self):
        """p1 서브 시 p1 득점 수."""
        return sum(1 for r in self.rounds if r.server == "player_1" and r.winner == "player_1")

    @property
    def p1_serve_total(self):
        return sum(1 for r in self.rounds if r.server == "player_1")

    @property
    def p2_serve_win(self):
        """p2 서브 시 p2 득점 수."""
        return sum(1 for r in self.rounds if r.server == "player_2" and r.winner == "player_2")

    @property
    def p2_serve_total(self):
        return sum(1 for r in self.rounds if r.server == "player_2")

    @property
    def avg_rally_length(self):
        if not self.rounds:
            return 0
        return sum(r.rally_length for r in self.rounds) / len(self.rounds)


MAX_RALLY_STEPS = 3000   # 라운드당 최대 스텝 (무한 랠리 방지)


def play_game_detailed(p1, p2, winning_score=15, seed=None):
    """상세 통계를 포함한 1판 수행. 라운드당 MAX_RALLY_STEPS 초과 시 무승부 처리."""
    is_p1_computer = p1.player_type == "builtin"
    is_p2_computer = p2.player_type == "builtin"

    env = raw_env(
        winning_score=winning_score,
        serve="winner",
        is_player1_computer=is_p1_computer,
        is_player2_computer=is_p2_computer,
    )
    env = SimplifyAction(env)
    env = NormalizeObservation(env)

    obs, info = env.reset(seed=seed)
    stats = GameStats()
    total_steps = 0
    rally_steps = 0
    current_server = "player_1"  # 첫 서브는 항상 p1
    truncated_rallies = 0

    while env.agents:
        actions = {
            "player_1": p1.get_action(obs.get("player_1"), env, "player_1"),
            "player_2": p2.get_action(obs.get("player_2"), env, "player_2"),
        }
        obs, rewards, terminated, truncated, infos = env.step(actions)
        rally_steps += 1
        total_steps += 1

        # 무한 랠리 감지 (라운드 단위)
        if rally_steps >= MAX_RALLY_STEPS:
            truncated_rallies += 1
            stats.rounds.append(RoundStats(
                server=current_server,
                winner="draw",
                rally_length=rally_steps,
            ))
            rally_steps = 0
            continue

        # 라운드 종료 감지 (보상이 0이 아니면 득점 발생)
        if rewards.get("player_1", 0) != 0:
            round_winner = "player_1" if rewards["player_1"] > 0 else "player_2"
            stats.rounds.append(RoundStats(
                server=current_server,
                winner=round_winner,
                rally_length=rally_steps,
            ))
            if round_winner == "player_1":
                stats.p1_score += 1
            else:
                stats.p2_score += 1
            current_server = round_winner
            rally_steps = 0

    env.close()
    stats.truncated_rallies = truncated_rallies
    stats.seed = seed
    return stats


def analyze_games(p1_spec, p2_spec, games=100, winning_score=15, seed=42):
    """여러 게임의 상세 통계를 집계."""
    p1 = make_player(p1_spec)
    p2 = make_player(p2_spec)
    rng = np.random.default_rng(seed)

    all_stats = []
    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        stats = play_game_detailed(p1, p2, winning_score=winning_score, seed=game_seed)
        all_stats.append(stats)

    # 집계
    p1_wins = sum(1 for s in all_stats if s.winner == "player_1")
    all_rounds = [r for s in all_stats for r in s.rounds]

    p1_serve_rounds = [r for r in all_rounds if r.server == "player_1"]
    p2_serve_rounds = [r for r in all_rounds if r.server == "player_2"]

    p1_serve_p1_win = sum(1 for r in p1_serve_rounds if r.winner == "player_1")
    p2_serve_p2_win = sum(1 for r in p2_serve_rounds if r.winner == "player_2")

    rally_lengths = [r.rally_length for r in all_rounds]

    print(f"=== {p1.name} (p1) vs {p2.name} (p2) — {games}판 {winning_score}점제 ===\n")
    print(f"승패: p1 {p1_wins}W {games - p1_wins}L ({p1_wins / games * 100:.0f}%)\n")

    print(f"--- 서브별 득점률 ---")
    if p1_serve_rounds:
        pct = p1_serve_p1_win / len(p1_serve_rounds) * 100
        print(f"p1 서브 시 p1 득점: {p1_serve_p1_win}/{len(p1_serve_rounds)} ({pct:.1f}%)")
    if p2_serve_rounds:
        pct = p2_serve_p2_win / len(p2_serve_rounds) * 100
        print(f"p2 서브 시 p2 득점: {p2_serve_p2_win}/{len(p2_serve_rounds)} ({pct:.1f}%)")

    print(f"\n--- 랠리 길이 ---")
    print(f"평균: {np.mean(rally_lengths):.1f} 스텝")
    print(f"중앙값: {np.median(rally_lengths):.1f} 스텝")
    print(f"최소/최대: {np.min(rally_lengths)}/{np.max(rally_lengths)} 스텝")

    return all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Game detailed stats")
    parser.add_argument("--p1", required=True)
    parser.add_argument("--p2", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--score", type=int, default=15)
    args = parser.parse_args()

    analyze_games(args.p1, args.p2, games=args.games, winning_score=args.score)
