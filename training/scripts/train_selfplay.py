"""Self-play 학습 스크립트 (PFSP + builtin 앵커).

p1_model(좌측)과 p2_model(우측)을 교대 학습.
상대 비율: latest_prob(최신) + builtin_prob(규칙AI) + 나머지(풀 PFSP).

Usage:
  uv run python training/scripts/train_selfplay.py --total-iterations 100 --steps-per-iter 20000
  uv run python training/scripts/train_selfplay.py --p1-init exp/001/model --p2-init exp/002/model
"""

import argparse

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import ConvertSingleAgent, NormalizeObservation, SimplifyAction
from training.utils.opponent_pool import OpponentPool, make_opponent_policy, make_builtin_policy
from training.utils.elo import make_player
from training.utils.match_stats import play_game_detailed


class GymnasiumWrapper(gym.Env):
    """ConvertSingleAgent를 gymnasium.Env으로 래핑."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space(env.side)
        self.action_space = env.action_space(env.side)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def make_selfplay_env(rank, side, seed=0, opponent_is_builtin=False):
    def _init():
        is_p1_computer = opponent_is_builtin and side == "player_1"
        is_p2_computer = opponent_is_builtin and side == "player_2"
        env = raw_env(
            winning_score=15, serve="winner",
            is_player1_computer=is_p1_computer,
            is_player2_computer=is_p2_computer,
        )
        env = SimplifyAction(env)
        env = NormalizeObservation(env)
        env = ConvertSingleAgent(env, side=side)
        env = GymnasiumWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def set_opponent_in_vecenv(vec_env, opponent_policy, is_builtin=False, side=None):
    """DummyVecEnv 내부 환경들의 opponent policy를 일괄 교체.

    builtin 상대 시 환경을 재생성하여 is_player*_computer를 설정.
    """
    for i, env_wrapper in enumerate(vec_env.envs):
        convert_env = env_wrapper.env  # GymnasiumWrapper -> ConvertSingleAgent

        # builtin 플래그가 변경되었으면 내부 환경의 computer 설정 업데이트
        inner_env = convert_env.env  # ConvertSingleAgent -> NormalizeObservation -> SimplifyAction -> raw_env
        # raw_env까지 탐색
        raw = inner_env
        while hasattr(raw, 'env'):
            raw = raw.env

        if side == "player_1":
            raw.is_player2_computer = is_builtin
        elif side == "player_2":
            raw.is_player1_computer = is_builtin

        if is_builtin:
            convert_env.set_opponent_policy(make_builtin_policy())
        else:
            convert_env.set_opponent_policy(opponent_policy)


def evaluate_selfplay_detailed(p1_path, p2_path, games=20, winning_score=15):
    """상세 통계 포함 평가."""
    matchups = {}

    # p1 vs p2
    p1 = make_player(p1_path)
    p2 = make_player(p2_path)
    rounds_all = []
    p1_wins = 0
    for _ in range(games):
        stats = play_game_detailed(p1, p2, winning_score=winning_score)
        p1_wins += 1 if stats.winner == "player_1" else 0
        rounds_all.extend(stats.rounds)
    matchups["p1_vs_p2"] = _summarize(p1_wins, games, rounds_all)

    # p1 vs baselines
    for opp_name in ("random", "builtin"):
        opp = make_player(opp_name)
        rounds_all = []
        wins = 0
        for _ in range(games):
            stats = play_game_detailed(p1, opp, winning_score=winning_score)
            wins += 1 if stats.winner == "player_1" else 0
            rounds_all.extend(stats.rounds)
        matchups[f"p1_vs_{opp_name}"] = _summarize(wins, games, rounds_all)

    # p2 vs baselines
    for opp_name in ("random", "builtin"):
        opp = make_player(opp_name)
        rounds_all = []
        wins = 0
        for _ in range(games):
            stats = play_game_detailed(opp, p2, winning_score=winning_score)
            wins += 1 if stats.winner == "player_2" else 0
            rounds_all.extend(stats.rounds)
        matchups[f"p2_vs_{opp_name}"] = _summarize(wins, games, rounds_all)

    return matchups


def _summarize(wins, games, rounds):
    """매치 통계 요약."""
    p1_serve = [r for r in rounds if r.server == "player_1"]
    p2_serve = [r for r in rounds if r.server == "player_2"]
    rally_lengths = [r.rally_length for r in rounds]

    return {
        "wins": wins,
        "losses": games - wins,
        "win_rate": wins / games,
        "p1_serve_win": sum(1 for r in p1_serve if r.winner == "player_1") / max(len(p1_serve), 1),
        "p2_serve_win": sum(1 for r in p2_serve if r.winner == "player_2") / max(len(p2_serve), 1),
        "avg_rally": np.mean(rally_lengths) if rally_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-play training (PFSP + builtin anchor)")
    parser.add_argument("--total-iterations", type=int, default=100)
    parser.add_argument("--steps-per-iter", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latest-prob", type=float, default=0.5)
    parser.add_argument("--builtin-prob", type=float, default=0.2)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--max-pool", type=int, default=50)
    parser.add_argument("--tensorboard-log", default="tensorboard_logs/selfplay")
    parser.add_argument("--save-dir", default="models/checkpoints")
    parser.add_argument("--p1-init", default=None, help="Pretrained p1 model path")
    parser.add_argument("--p2-init", default=None, help="Pretrained p2 model path")
    args = parser.parse_args()

    pool_prob = 1.0 - args.latest_prob - args.builtin_prob
    assert pool_prob >= 0, "latest_prob + builtin_prob must be <= 1.0"

    # 환경 생성 (DummyVecEnv)
    p1_envs = DummyVecEnv([make_selfplay_env(i, "player_1", args.seed) for i in range(args.num_envs)])
    p2_envs = DummyVecEnv([make_selfplay_env(i, "player_2", args.seed + 100) for i in range(args.num_envs)])

    # 모델 초기화
    if args.p1_init:
        p1_model = PPO.load(args.p1_init, env=p1_envs, device="cpu", seed=args.seed)
        print(f"Loaded p1 from {args.p1_init}")
    else:
        p1_model = PPO("MlpPolicy", p1_envs, device="cpu", verbose=0, seed=args.seed)

    if args.p2_init:
        p2_model = PPO.load(args.p2_init, env=p2_envs, device="cpu", seed=args.seed + 1)
        print(f"Loaded p2 from {args.p2_init}")
    else:
        p2_model = PPO("MlpPolicy", p2_envs, device="cpu", verbose=0, seed=args.seed + 1)

    # TensorBoard 로거
    p1_logger = configure(f"{args.tensorboard_log}/p1", ["tensorboard", "stdout"])
    p2_logger = configure(f"{args.tensorboard_log}/p2", ["tensorboard", "stdout"])
    p1_model.set_logger(p1_logger)
    p2_model.set_logger(p2_logger)

    # 상대풀
    pool_p1 = OpponentPool("models/pool/p1", "p1", max_pool_size=args.max_pool)
    pool_p2 = OpponentPool("models/pool/p2", "p2", max_pool_size=args.max_pool)

    print(f"Self-play training: {args.total_iterations} iterations x {args.steps_per_iter} steps")
    print(f"Envs: {args.num_envs} (DummyVecEnv)")
    print(f"Opponent mix: latest={args.latest_prob}, builtin={args.builtin_prob}, pool(PFSP)={pool_prob:.1f}")

    for iteration in range(args.total_iterations):
        # --- Train p1 against p2 opponent ---
        opp_model, opp_name, is_builtin = pool_p2.sample_opponent(
            p2_model, args.latest_prob, args.builtin_prob)
        if is_builtin:
            set_opponent_in_vecenv(p1_envs, None, is_builtin=True, side="player_1")
        else:
            set_opponent_in_vecenv(p1_envs, make_opponent_policy(opp_model), is_builtin=False, side="player_1")
        p1_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)

        # --- Train p2 against p1 opponent ---
        opp_model, opp_name, is_builtin = pool_p1.sample_opponent(
            p1_model, args.latest_prob, args.builtin_prob)
        if is_builtin:
            set_opponent_in_vecenv(p2_envs, None, is_builtin=True, side="player_2")
        else:
            set_opponent_in_vecenv(p2_envs, make_opponent_policy(opp_model), is_builtin=False, side="player_2")
        p2_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)

        # --- Save to pool ---
        if iteration % args.save_interval == 0 and iteration > 0:
            pool_p1.add_checkpoint(p1_model, iteration)
            pool_p2.add_checkpoint(p2_model, iteration)

        # --- Evaluate ---
        if iteration % args.eval_freq == 0:
            p1_model.save(f"{args.save_dir}/p1/selfplay_latest")
            p2_model.save(f"{args.save_dir}/p2/selfplay_latest")

            matchups = evaluate_selfplay_detailed(
                f"{args.save_dir}/p1/selfplay_latest",
                f"{args.save_dir}/p2/selfplay_latest",
                games=args.eval_games,
                winning_score=15,
            )

            # 출력 + 로깅
            total_steps = (iteration + 1) * args.steps_per_iter * 2
            print(f"\n[Iter {iteration}/{args.total_iterations}, {total_steps} total steps]")
            for match, s in matchups.items():
                print(f"  {match}: {s['wins']}W {s['losses']}L ({s['win_rate']*100:.0f}%)"
                      f"  서브: p1={s['p1_serve_win']*100:.0f}% p2={s['p2_serve_win']*100:.0f}%"
                      f"  랠리: {s['avg_rally']:.0f}")
                p1_logger.record(f"eval/{match}_winrate", s["win_rate"])
                p1_logger.record(f"eval/{match}_avg_rally", s["avg_rally"])
            p1_logger.dump(step=total_steps)

    # 최종 모델 저장
    p1_model.save(f"{args.save_dir}/p1/selfplay_final")
    p2_model.save(f"{args.save_dir}/p2/selfplay_final")
    print(f"\nTraining complete. Models saved to {args.save_dir}/p1/ and {args.save_dir}/p2/")

    p1_envs.close()
    p2_envs.close()


if __name__ == "__main__":
    main()
