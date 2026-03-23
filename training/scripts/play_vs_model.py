"""학습된 모델과 직접 대전하는 스크립트.

조작법 (Player 2, 우측):
  방향키: 이동/점프
  Z키: 파워히트

Usage:
  uv run python training/scripts/play_vs_model.py --model models/checkpoints/ppo_vs_builtin
"""

import argparse

import pygame
from stable_baselines3 import PPO

from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import NormalizeObservation, SimplifyAction

# SimplifyAction의 player_2 action_map: (0, 1, 2, 4, 3, 7, 6, 10, 12, 11, 13, 15, 17)
# 인덱스 → 의미 (player_2 기준, 상대적 방향):
#  0: NOOP
#  1: POWER_HIT
#  2: UP
#  3: FORWARD (toward net = LEFT for p2)
#  4: BACKWARD (away from net = RIGHT for p2)
#  5: FORWARD+UP
#  6: BACKWARD+UP
#  7: POWER_HIT+UP
#  8: POWER_HIT+FORWARD
#  9: POWER_HIT+BACKWARD
# 10: POWER_HIT+DOWN
# 11: POWER_HIT+FORWARD+UP
# 12: POWER_HIT+BACKWARD+DOWN


def get_human_action():
    """현재 눌린 키를 읽어서 SimplifyAction 기준 player_2 action 반환."""
    keys = pygame.key.get_pressed()

    up = keys[pygame.K_UP]
    down = keys[pygame.K_DOWN]
    fwd = keys[pygame.K_LEFT]    # player_2는 좌측이 forward(네트 쪽)
    bwd = keys[pygame.K_RIGHT]   # player_2는 우측이 backward
    hit = keys[pygame.K_z]

    if hit:
        if fwd and up:    return 11
        if bwd and down:  return 12
        if up:            return 7
        if fwd:           return 8
        if bwd:           return 9
        if down:          return 10
        return 1
    else:
        if fwd and up:    return 5
        if bwd and up:    return 6
        if up:            return 2
        if fwd:           return 3
        if bwd:           return 4
        if down:          return 0  # down only = NOOP
        return 0


def main():
    parser = argparse.ArgumentParser(description="Play against a trained model")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--score", type=int, default=15)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")

    env = raw_env(winning_score=args.score, serve="winner", render_mode="human")
    env = SimplifyAction(env)
    env = NormalizeObservation(env)

    clock = pygame.time.Clock()
    obs, info = env.reset(seed=42)

    while env.agents:
        env.render()
        clock.tick(25)

        pygame.event.pump()
        human_action = get_human_action()

        p1_action, _ = model.predict(obs["player_1"], deterministic=True)
        actions = {"player_1": int(p1_action), "player_2": human_action}

        obs, rewards, terminated, truncated, info = env.step(actions)

    env.render()
    pygame.time.wait(2000)
    env.close()


if __name__ == "__main__":
    main()
