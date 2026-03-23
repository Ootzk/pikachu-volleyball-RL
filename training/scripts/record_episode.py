"""pika-zoo 에피소드 녹화 스크립트.

Usage:
  # 양쪽 랜덤
  uv run python training/scripts/record_episode.py

  # player_1에 학습된 모델 사용
  uv run python training/scripts/record_episode.py --p1-model models/checkpoints/ppo_pikazoo
"""

import argparse
import sys

sys.path.insert(0, "training/env")

from moviepy import ImageSequenceClip
from stable_baselines3 import PPO

from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import NormalizeObservation, SimplifyAction


def record_episode(env, output_path, seed=None, fps=25, p1_model=None, p2_model=None):
    """에피소드를 실행하고 영상으로 저장.

    Args:
        env: SimplifyAction + NormalizeObservation 래퍼가 적용된 환경.
        output_path: 저장할 영상 경로.
        seed: 환경 시드.
        fps: 영상 프레임레이트.
        p1_model: player_1 모델 (None이면 랜덤).
        p2_model: player_2 모델 (None이면 랜덤).
    """
    obs, info = env.reset(seed=seed)

    frames = []
    while env.agents:
        frames.append(env.render())

        actions = {}
        for agent, model in [("player_1", p1_model), ("player_2", p2_model)]:
            if agent not in env.agents:
                continue
            if model is not None:
                action, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = int(action)
            else:
                actions[agent] = env.action_space(agent).sample()

        obs, rewards, terminated, truncated, info = env.step(actions)
    frames.append(env.render())
    env.close()

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, logger="bar")
    print(f"\nSaved to {output_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Record a pika-zoo episode")
    parser.add_argument("-o", "--output", default="training/scripts/episode.mp4")
    parser.add_argument("--score", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--p1-model", default=None, help="Path to player_1 model")
    parser.add_argument("--p2-model", default=None, help="Path to player_2 model")
    args = parser.parse_args()

    env = raw_env(winning_score=args.score, serve="winner", render_mode="rgb_array")
    env = SimplifyAction(env)
    env = NormalizeObservation(env)

    p1_model = PPO.load(args.p1_model) if args.p1_model else None
    p2_model = PPO.load(args.p2_model) if args.p2_model else None

    record_episode(env, args.output, seed=args.seed, fps=args.fps,
                   p1_model=p1_model, p2_model=p2_model)


if __name__ == "__main__":
    main()
