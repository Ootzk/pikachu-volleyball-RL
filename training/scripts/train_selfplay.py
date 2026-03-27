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


def _run_matchup(name, p1_spec, p2_spec, games, winning_score, perspective, seed):
    """단일 매치업 평가 (spawn 워커에서 실행 가능 — pickle 가능한 인자만 받음)."""
    rng = np.random.default_rng(seed)
    p1_player = make_player(p1_spec)
    p2_player = make_player(p2_spec)
    rounds_all = []
    all_stats = []
    wins = 0
    truncated_total = 0
    for i in range(games):
        game_seed = int(rng.integers(0, 2**31))
        stats = play_game_detailed(p1_player, p2_player, winning_score=winning_score, seed=game_seed)
        all_stats.append(stats)
        if stats.truncated_rallies > 0:
            truncated_total += stats.truncated_rallies
        if perspective == "p1":
            wins += 1 if stats.winner == "player_1" else 0
        else:
            wins += 1 if stats.winner == "player_2" else 0
        rounds_all.extend(stats.rounds)
    summary = _summarize(wins, games, rounds_all, all_stats, perspective)
    summary["truncated_rallies"] = truncated_total
    return name, summary


def evaluate_selfplay_detailed(p1_model, p2_model, games=20, winning_score=15, seed=42, save_dir=None):
    """상세 통계 포함 평가. 5매치업 병렬 실행 (spawn)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # eval 전에 save_dir에 저장된 모델 경로 사용
    assert save_dir is not None, "save_dir required for parallel eval"
    p1_path = f"{save_dir}/p1/selfplay_latest"
    p2_path = f"{save_dir}/p2/selfplay_latest"

    rng = np.random.default_rng(seed)
    matchup_defs = [
        ("p1_vs_p2", p1_path, p2_path, "p1"),
        ("p1_vs_random", p1_path, "random", "p1"),
        ("p1_vs_builtin", p1_path, "builtin", "p1"),
        ("p2_vs_random", "random", p2_path, "p2"),
        ("p2_vs_builtin", "builtin", p2_path, "p2"),
    ]
    matchup_seeds = [int(rng.integers(0, 2**31)) for _ in matchup_defs]

    ctx = mp.get_context("spawn")
    matchups = {}
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as executor:
        futures = {}
        for (name, p1_spec, p2_spec, perspective), mseed in zip(matchup_defs, matchup_seeds):
            fut = executor.submit(_run_matchup, name, p1_spec, p2_spec, games, winning_score, perspective, mseed)
            futures[fut] = name
        for fut in futures:
            name, summary = fut.result()
            matchups[name] = summary

    return matchups


def _summarize(wins, games, rounds, all_stats, perspective):
    """매치 통계 요약."""
    p1_serve = [r for r in rounds if r.server == "player_1"]
    p2_serve = [r for r in rounds if r.server == "player_2"]
    rally_lengths = [r.rally_length for r in rounds]

    if perspective == "p1":
        avg_score = np.mean([s.p1_score for s in all_stats]) if all_stats else 0
        avg_opp_score = np.mean([s.p2_score for s in all_stats]) if all_stats else 0
    else:
        avg_score = np.mean([s.p2_score for s in all_stats]) if all_stats else 0
        avg_opp_score = np.mean([s.p1_score for s in all_stats]) if all_stats else 0

    return {
        "wins": wins,
        "losses": games - wins,
        "win_rate": wins / games,
        "avg_score": float(avg_score),
        "avg_opp_score": float(avg_opp_score),
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
    parser.add_argument("--curriculum", default=None,
                        help="Path to curriculum JSON file for dynamic opponent mix")
    parser.add_argument("--adaptive", default=None,
                        help="Path to adaptive curriculum JSON file")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--tensorboard-log", default="tensorboard_logs/selfplay")
    parser.add_argument("--save-dir", required=True, help="Directory for checkpoints and opponent pool")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    parser.add_argument("--p1-init", default=None, help="Pretrained p1 model path")
    parser.add_argument("--p2-init", default=None, help="Pretrained p2 model path")
    args = parser.parse_args()

    # 커리큘럼 로드
    curriculum_schedule = None
    if args.curriculum:
        import json
        with open(args.curriculum) as f:
            curriculum_schedule = json.load(f)["schedule"]
        curriculum_schedule.sort(key=lambda x: x["iter"])

    # Adaptive 커리큘럼 로드
    adaptive_config = None
    if args.adaptive:
        import json
        with open(args.adaptive) as f:
            adaptive_config = json.load(f)

    p1_builtin_winrate = 0.0  # adaptive 모드용
    p2_builtin_winrate = 0.0

    def _adaptive_probs(winrate):
        """adaptive 승률 기반 비율 계산."""
        thresholds = adaptive_config["thresholds"]
        if winrate <= thresholds[0]["winrate"]:
            return thresholds[0]["latest"], thresholds[0]["builtin"]
        if winrate >= thresholds[-1]["winrate"]:
            return thresholds[-1]["latest"], thresholds[-1]["builtin"]
        for i in range(len(thresholds) - 1):
            a, b = thresholds[i], thresholds[i + 1]
            if a["winrate"] <= winrate <= b["winrate"]:
                t = (winrate - a["winrate"]) / (b["winrate"] - a["winrate"])
                builtin = a["builtin"] + t * (b["builtin"] - a["builtin"])
                latest = a["latest"] + t * (b["latest"] - a["latest"])
                return latest, builtin
        return thresholds[-1]["latest"], thresholds[-1]["builtin"]

    def get_probs(iteration, side="p1"):
        """커리큘럼 또는 고정 비율 반환: (latest_prob, builtin_prob)."""
        if adaptive_config:
            wr = p1_builtin_winrate if side == "p1" else p2_builtin_winrate
            return _adaptive_probs(wr)
            return thresholds[-1]["latest"], thresholds[-1]["builtin"]

        if curriculum_schedule is None:
            return args.latest_prob, args.builtin_prob
        # 선형 보간
        if iteration <= curriculum_schedule[0]["iter"]:
            s = curriculum_schedule[0]
            return s["latest"], s["builtin"]
        if iteration >= curriculum_schedule[-1]["iter"]:
            s = curriculum_schedule[-1]
            return s["latest"], s["builtin"]
        for i in range(len(curriculum_schedule) - 1):
            a, b = curriculum_schedule[i], curriculum_schedule[i + 1]
            if a["iter"] <= iteration <= b["iter"]:
                t = (iteration - a["iter"]) / (b["iter"] - a["iter"])
                latest = a["latest"] + t * (b["latest"] - a["latest"])
                builtin = a["builtin"] + t * (b["builtin"] - a["builtin"])
                return latest, builtin
        return args.latest_prob, args.builtin_prob

    initial_latest, initial_builtin = get_probs(0)
    pool_prob = 1.0 - initial_latest - initial_builtin
    assert pool_prob >= 0, "latest_prob + builtin_prob must be <= 1.0"

    # 환경 생성 (DummyVecEnv)
    p1_envs = DummyVecEnv([make_selfplay_env(i, "player_1", args.seed) for i in range(args.num_envs)])
    p2_envs = DummyVecEnv([make_selfplay_env(i, "player_2", args.seed + 100) for i in range(args.num_envs)])

    # 모델 초기화
    ppo_kwargs = dict(device="cpu", verbose=0, ent_coef=args.ent_coef)
    if args.p1_init:
        p1_model = PPO.load(args.p1_init, env=p1_envs, seed=args.seed, **ppo_kwargs)
        print(f"Loaded p1 from {args.p1_init}")
    else:
        p1_model = PPO("MlpPolicy", p1_envs, seed=args.seed, **ppo_kwargs)

    if args.p2_init:
        p2_model = PPO.load(args.p2_init, env=p2_envs, seed=args.seed + 1, **ppo_kwargs)
        print(f"Loaded p2 from {args.p2_init}")
    else:
        p2_model = PPO("MlpPolicy", p2_envs, seed=args.seed + 1, **ppo_kwargs)

    # TensorBoard 로거
    p1_logger = configure(f"{args.tensorboard_log}/p1", ["tensorboard", "stdout"])
    p2_logger = configure(f"{args.tensorboard_log}/p2", ["tensorboard", "stdout"])
    p1_model.set_logger(p1_logger)
    p2_model.set_logger(p2_logger)


    # 상대풀 (save_dir 내 checkpoints에 통합)
    pool_p1 = OpponentPool(f"{args.save_dir}/p1", "p1")
    pool_p2 = OpponentPool(f"{args.save_dir}/p2", "p2")

    print(f"Self-play training: {args.total_iterations} iterations x {args.steps_per_iter} steps")
    print(f"Envs: {args.num_envs} (DummyVecEnv)")
    if adaptive_config:
        first, last = adaptive_config["thresholds"][0], adaptive_config["thresholds"][-1]
        print(f"Adaptive curriculum: builtin {first['builtin']*100:.0f}%→{last['builtin']*100:.0f}% based on win rate")
    elif curriculum_schedule:
        first, last = curriculum_schedule[0], curriculum_schedule[-1]
        print(f"Curriculum: builtin {first['builtin']*100:.0f}%→{last['builtin']*100:.0f}%, "
              f"latest {first['latest']*100:.0f}%→{last['latest']*100:.0f}%")
    else:
        print(f"Opponent mix: latest={args.latest_prob}, builtin={args.builtin_prob}, pool(PFSP)={pool_prob:.1f}")

    best_p1_builtin = -1.0
    best_p2_builtin = -1.0

    for iteration in range(args.total_iterations):
        # --- Evaluate ---
        if iteration % args.eval_freq == 0:
            p1_model.save(f"{args.save_dir}/p1/selfplay_latest")
            p2_model.save(f"{args.save_dir}/p2/selfplay_latest")
            matchups = evaluate_selfplay_detailed(
                p1_model, p2_model,
                games=args.eval_games,
                winning_score=15,
                save_dir=args.save_dir,
            )

            step = p1_model.num_timesteps
            print(f"\n[Iter {iteration}/{args.total_iterations}, p1_step={step}]", flush=True)
            for match, s in matchups.items():
                print(f"  {match}: {s['wins']}W {s['losses']}L ({s['win_rate']*100:.0f}%)"
                      f"  득점: {s['avg_score']:.1f}-{s['avg_opp_score']:.1f}"
                      f"  서브: p1={s['p1_serve_win']*100:.0f}% p2={s['p2_serve_win']*100:.0f}%"
                      f"  랠리: {s['avg_rally']:.0f}", flush=True)

                # p1 관점 매치 → p1_logger
                if match.startswith("p1_vs_"):
                    opponent = match[len("p1_vs_"):]
                    p1_logger.record(f"eval/vs_{opponent}_winrate", s["win_rate"])
                    p1_logger.record(f"eval/vs_{opponent}_avg_score", s["avg_score"])
                    p1_logger.record(f"eval/vs_{opponent}_avg_rally", s["avg_rally"])

                # p2 관점 매치 → p2_logger
                if match.startswith("p2_vs_"):
                    opponent = match[len("p2_vs_"):]
                    p2_logger.record(f"eval/vs_{opponent}_winrate", s["win_rate"])
                    p2_logger.record(f"eval/vs_{opponent}_avg_score", s["avg_score"])
                    p2_logger.record(f"eval/vs_{opponent}_avg_rally", s["avg_rally"])

                # p1_vs_p2는 양쪽 관점 모두 기록
                if match == "p1_vs_p2":
                    p2_logger.record("eval/vs_p1_winrate", 1.0 - s["win_rate"])
                    p2_logger.record("eval/vs_p1_avg_score", s["avg_opp_score"])
                    p2_logger.record("eval/vs_p1_avg_rally", s["avg_rally"])

            p1_logger.dump(step=step)
            p2_logger.dump(step=step)

            # Adaptive 커리큘럼 업데이트
            p1_wr = matchups.get("p1_vs_builtin", {}).get("win_rate", 0)
            p2_wr = matchups.get("p2_vs_builtin", {}).get("win_rate", 0)
            if adaptive_config:
                p1_builtin_winrate = p1_wr
                p2_builtin_winrate = p2_wr
                p1_latest, p1_builtin = get_probs(iteration, side="p1")
                p2_latest, p2_builtin = get_probs(iteration, side="p2")
                print(f"  [ADAPTIVE] p1: wr={p1_wr*100:.0f}% → builtin={p1_builtin*100:.0f}%"
                      f"  |  p2: wr={p2_wr*100:.0f}% → builtin={p2_builtin*100:.0f}%", flush=True)

            # Best model 저장
            if p1_wr > best_p1_builtin:
                best_p1_builtin = p1_wr
                p1_model.save(f"{args.save_dir}/p1/selfplay_best")
                print(f"  [BEST] p1 vs builtin: {p1_wr*100:.0f}% (iter {iteration})", flush=True)
            if p2_wr > best_p2_builtin:
                best_p2_builtin = p2_wr
                p2_model.save(f"{args.save_dir}/p2/selfplay_best")
                print(f"  [BEST] p2 vs builtin: {p2_wr*100:.0f}% (iter {iteration})", flush=True)

        # --- Train ---
        p1_latest_prob, p1_builtin_prob = get_probs(iteration, side="p1")
        p2_latest_prob, p2_builtin_prob = get_probs(iteration, side="p2")

        # 커리큘럼 메타데이터 로깅
        p1_logger.record("curriculum/builtin_prob", p1_builtin_prob)
        p1_logger.record("curriculum/latest_prob", p1_latest_prob)
        p1_logger.record("curriculum/pool_prob", 1.0 - p1_builtin_prob - p1_latest_prob)
        p2_logger.record("curriculum/builtin_prob", p2_builtin_prob)
        p2_logger.record("curriculum/latest_prob", p2_latest_prob)
        p2_logger.record("curriculum/pool_prob", 1.0 - p2_builtin_prob - p2_latest_prob)

        # Train p1 against p2 opponent
        opp_model, opp_name, is_builtin = pool_p2.sample_opponent(
            p2_model, p1_latest_prob, p1_builtin_prob)
        if is_builtin:
            set_opponent_in_vecenv(p1_envs, None, is_builtin=True, side="player_1")
        else:
            set_opponent_in_vecenv(p1_envs, make_opponent_policy(opp_model), is_builtin=False, side="player_1")
        p1_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)

        # Train p2 against p1 opponent
        opp_model, opp_name, is_builtin = pool_p1.sample_opponent(
            p1_model, p2_latest_prob, p2_builtin_prob)
        if is_builtin:
            set_opponent_in_vecenv(p2_envs, None, is_builtin=True, side="player_2")
        else:
            set_opponent_in_vecenv(p2_envs, make_opponent_policy(opp_model), is_builtin=False, side="player_2")
        p2_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)

        # --- Save to pool ---
        if iteration % args.save_interval == 0 and iteration > 0:
            pool_p1.add_checkpoint(p1_model, iteration)
            pool_p2.add_checkpoint(p2_model, iteration)

    # 최종 모델 저장
    p1_model.save(f"{args.save_dir}/p1/selfplay_final")
    p2_model.save(f"{args.save_dir}/p2/selfplay_final")
    print(f"\nTraining complete. Models saved to {args.save_dir}/p1/ and {args.save_dir}/p2/")

    p1_envs.close()
    p2_envs.close()


if __name__ == "__main__":
    main()
