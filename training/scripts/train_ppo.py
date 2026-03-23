"""SB3 PPO 학습 스크립트.

Usage:
  # 랜덤 상대 학습
  uv run python training/scripts/train_ppo.py --timesteps 1000000

  # 규칙 AI 상대 학습 + ELO 평가
  uv run python training/scripts/train_ppo.py --opponent builtin --timesteps 1000000 --eval-freq 50000
"""

import argparse
import os
import sys

sys.path.insert(0, "training/utils")
sys.path.insert(0, "training/env")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from elo import evaluate_model
from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import ConvertSingleAgent, NormalizeObservation, SimplifyAction


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


class EloEvalCallback(BaseCallback):
    """학습 중 주기적으로 ELO를 평가하는 콜백."""

    def __init__(self, eval_freq, save_path, eval_games=20, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.eval_games = eval_games

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # 임시 모델 저장 후 평가
            tmp_path = self.save_path + "_tmp"
            self.model.save(tmp_path)

            results, elo = evaluate_model(
                tmp_path, opponents=("random", "builtin"),
                games=self.eval_games, winning_score=5,
            )

            # TensorBoard 로깅
            self.logger.record("eval/elo", elo)
            for opp_name, (wins, losses) in results.items():
                win_pct = wins / (wins + losses)
                self.logger.record(f"eval/win_rate_{opp_name}", win_pct)

            if self.verbose:
                print(f"\n[Eval @ {self.num_timesteps} steps] ELO: {elo:.0f}")
                for opp_name, (wins, losses) in results.items():
                    print(f"  vs {opp_name}: {wins}W {losses}L")

            # 임시 파일 정리
            os.remove(tmp_path + ".zip")

        return True


def make_env(rank, seed=0, opponent="random"):
    """환경 팩토리 함수."""

    def _init():
        is_p2_computer = opponent == "builtin"
        env = raw_env(
            winning_score=15, serve="winner",
            is_player2_computer=is_p2_computer,
        )
        env = SimplifyAction(env)
        env = NormalizeObservation(env)
        env = ConvertSingleAgent(env, side="player_1")
        env = GymnasiumWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on pikazoo")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--save-path", default="models/checkpoints/ppo_pikazoo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random", choices=["random", "builtin"],
                        help="Opponent type: random or builtin rule AI")
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="ELO evaluation frequency in steps (0=disabled)")
    parser.add_argument("--tensorboard-log", default=None,
                        help="TensorBoard log directory")
    args = parser.parse_args()

    env = SubprocVecEnv([
        make_env(i, seed=args.seed, opponent=args.opponent)
        for i in range(args.num_envs)
    ])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device="cpu",
        tensorboard_log=args.tensorboard_log,
    )

    callbacks = []
    if args.eval_freq > 0:
        callbacks.append(EloEvalCallback(
            eval_freq=args.eval_freq // args.num_envs,  # SubprocVecEnv는 n_envs 스텝씩 진행
            save_path=args.save_path,
        ))

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    env.close()


if __name__ == "__main__":
    main()
