"""Self-play 학습 스크립트.

p1_model(좌측)과 p2_model(우측)을 교대 학습.

Usage:
  uv run python training/scripts/train_selfplay.py --total-iterations 100 --steps-per-iter 20000
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import ConvertSingleAgent, NormalizeObservation, SimplifyAction
from training.utils.opponent_pool import OpponentPool, make_opponent_policy
from training.utils.elo import make_player, play_game


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


def make_selfplay_env(rank, side, seed=0):
    def _init():
        env = raw_env(winning_score=15, serve="winner")
        env = SimplifyAction(env)
        env = NormalizeObservation(env)
        env = ConvertSingleAgent(env, side=side)
        env = GymnasiumWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def set_opponent_in_vecenv(vec_env, opponent_policy):
    """DummyVecEnv 내부 환경들의 opponent policy를 일괄 교체."""
    for env in vec_env.envs:
        env.env.set_opponent_policy(opponent_policy)


def evaluate_selfplay(p1_model, p2_model, games=20, winning_score=5):
    """p1 vs p2 직접 대전 + 각각 vs random/builtin 평가."""
    results = {}

    # p1 vs p2
    p1 = make_player(p1_model)
    p2 = make_player(p2_model)
    wins = sum(play_game(p1, p2, winning_score=winning_score) for _ in range(games))
    results["p1_vs_p2"] = (wins, games - wins)

    # p1 vs baselines
    for opp_name in ("random", "builtin"):
        opp = make_player(opp_name)
        wins = sum(play_game(p1, opp, winning_score=winning_score) for _ in range(games))
        results[f"p1_vs_{opp_name}"] = (wins, games - wins)

    # p2 vs baselines (p2는 우측이므로 상대가 p1 자리)
    for opp_name in ("random", "builtin"):
        opp = make_player(opp_name)
        wins = sum(play_game(opp, p2, winning_score=winning_score) for _ in range(games))
        results[f"p2_vs_{opp_name}"] = (games - wins, wins)

    return results


def main():
    parser = argparse.ArgumentParser(description="Self-play training")
    parser.add_argument("--total-iterations", type=int, default=100)
    parser.add_argument("--steps-per-iter", type=int, default=20000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latest-prob", type=float, default=0.8)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--max-pool", type=int, default=50)
    parser.add_argument("--tensorboard-log", default="tensorboard_logs/selfplay")
    parser.add_argument("--save-dir", default="models/checkpoints")
    parser.add_argument("--p1-init", default=None, help="Pretrained p1 model path")
    parser.add_argument("--p2-init", default=None, help="Pretrained p2 model path")
    args = parser.parse_args()

    # 환경 생성 (DummyVecEnv)
    p1_envs = DummyVecEnv([make_selfplay_env(i, "player_1", args.seed) for i in range(args.num_envs)])
    p2_envs = DummyVecEnv([make_selfplay_env(i, "player_2", args.seed + 100) for i in range(args.num_envs)])

    # 모델 초기화 (pretrained 모델이 있으면 로드)
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
    print(f"Envs: {args.num_envs} (DummyVecEnv), Latest prob: {args.latest_prob}")

    for iteration in range(args.total_iterations):
        # --- Train p1 against p2 opponent ---
        opp_model, opp_name = pool_p2.sample_opponent(p2_model, args.latest_prob)
        set_opponent_in_vecenv(p1_envs, make_opponent_policy(opp_model))
        p1_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)

        # --- Train p2 against p1 opponent ---
        opp_model, opp_name = pool_p1.sample_opponent(p1_model, args.latest_prob)
        set_opponent_in_vecenv(p2_envs, make_opponent_policy(opp_model))
        p2_model.learn(total_timesteps=args.steps_per_iter, reset_num_timesteps=False)

        # --- Save to pool ---
        if iteration % args.save_interval == 0 and iteration > 0:
            pool_p1.add_checkpoint(p1_model, iteration)
            pool_p2.add_checkpoint(p2_model, iteration)

        # --- Evaluate ---
        if iteration % args.eval_freq == 0:
            p1_model.save(f"{args.save_dir}/p1/selfplay_latest")
            p2_model.save(f"{args.save_dir}/p2/selfplay_latest")

            results = evaluate_selfplay(
                f"{args.save_dir}/p1/selfplay_latest",
                f"{args.save_dir}/p2/selfplay_latest",
                games=args.eval_games,
                winning_score=5,
            )

            # 로깅
            total_steps = (iteration + 1) * args.steps_per_iter * 2
            print(f"\n[Iter {iteration}/{args.total_iterations}, {total_steps} total steps]")
            for match, (w, l) in results.items():
                pct = w / (w + l) * 100
                print(f"  {match}: {w}W {l}L ({pct:.0f}%)")
                p1_logger.record(f"eval/{match}_winrate", w / (w + l))
            p1_logger.dump(step=total_steps)

    # 최종 모델 저장
    p1_model.save(f"{args.save_dir}/p1/selfplay_final")
    p2_model.save(f"{args.save_dir}/p2/selfplay_final")
    print(f"\nTraining complete. Models saved to {args.save_dir}/p1/ and {args.save_dir}/p2/")

    p1_envs.close()
    p2_envs.close()


if __name__ == "__main__":
    main()
