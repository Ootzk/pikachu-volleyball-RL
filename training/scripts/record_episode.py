"""pika-zoo 에피소드 녹화 스크립트."""

import argparse
import sys

sys.path.insert(0, "training/env")

from moviepy import ImageSequenceClip
from pikazoo.env.pikazoo_env import raw_env


def record_episode(env, output_path, seed=None, fps=25):
    """env를 랜덤 행동으로 한 에피소드 실행하고 영상으로 저장."""
    obs, info = env.reset(seed=seed)

    frames = []
    while env.agents:
        frames.append(env.render())
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
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
    args = parser.parse_args()

    env = raw_env(winning_score=args.score, serve="winner", render_mode="rgb_array")
    record_episode(env, args.output, seed=args.seed, fps=args.fps)


if __name__ == "__main__":
    main()
