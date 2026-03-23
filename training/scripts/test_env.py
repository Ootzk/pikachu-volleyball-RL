"""pika-zoo 환경 동작 확인 스크립트."""

import sys
sys.path.insert(0, "training/env")

from pikazoo.env.pikazoo_env import raw_env


def main():
    env = raw_env(winning_score=5, serve="winner")

    # 공간 확인
    print("=== Environment Spaces ===")
    for agent in env.possible_agents:
        print(f"  {agent}:")
        print(f"    observation: {env.observation_space(agent)}")
        print(f"    action:      {env.action_space(agent)}")

    # 에피소드 실행
    obs, info = env.reset(seed=42)
    print(f"\n=== Episode Start ===")
    print(f"Agents: {env.agents}")
    print(f"Observation shapes: { {k: v.shape for k, v in obs.items()} }")

    step = 0
    total_rewards = {agent: 0.0 for agent in env.possible_agents}

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)

        for agent in rewards:
            total_rewards[agent] += rewards[agent]
        step += 1

    print(f"\n=== Episode Done ===")
    print(f"Steps: {step}")
    print(f"Total rewards: {total_rewards}")

    env.close()
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
