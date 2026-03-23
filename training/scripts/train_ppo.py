"""SB3 PPO 기본 학습 스크립트."""

import argparse
import sys

sys.path.insert(0, "training/env")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

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


def make_env(rank, seed=0):
    """환경 팩토리 함수."""

    def _init():
        env = raw_env(winning_score=15, serve="winner")
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
    args = parser.parse_args()

    env = SubprocVecEnv([make_env(i, seed=args.seed) for i in range(args.num_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    env.close()


if __name__ == "__main__":
    main()
