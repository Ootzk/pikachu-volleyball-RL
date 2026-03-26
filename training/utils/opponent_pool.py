"""Self-play 상대풀 관리."""

import os
import random

from stable_baselines3 import PPO


def make_opponent_policy(model):
    """SB3 모델을 callable (obs) -> action으로 변환."""
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


class OpponentPool:
    """과거 체크포인트를 관리하고 상대를 샘플링."""

    def __init__(self, pool_dir, side, max_pool_size=50):
        self.pool_dir = pool_dir
        self.side = side
        self.max_pool_size = max_pool_size
        self.checkpoints = []
        os.makedirs(pool_dir, exist_ok=True)

    def add_checkpoint(self, model, iteration):
        path = os.path.join(self.pool_dir, f"{self.side}_iter{iteration:06d}")
        model.save(path)
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_pool_size:
            self._prune()
        return path

    def sample_opponent(self, latest_model, latest_prob=0.8):
        """상대 모델 반환. latest_prob 확률로 최신 모델, 나머지는 풀에서 샘플링."""
        if not self.checkpoints or random.random() < latest_prob:
            return latest_model, "latest"
        path = random.choice(self.checkpoints)
        model = PPO.load(path, device="cpu")
        return model, os.path.basename(path)

    def _prune(self):
        """max_pool_size 초과 시 오래된 체크포인트 제거. 첫 번째와 마지막은 유지."""
        if len(self.checkpoints) <= self.max_pool_size:
            return
        keep = set()
        keep.add(0)
        keep.add(len(self.checkpoints) - 1)
        step = max(1, len(self.checkpoints) // self.max_pool_size)
        for i in range(0, len(self.checkpoints), step):
            keep.add(i)
        new_checkpoints = []
        for i, path in enumerate(self.checkpoints):
            if i in keep and len(new_checkpoints) < self.max_pool_size:
                new_checkpoints.append(path)
            else:
                if os.path.exists(path + ".zip"):
                    os.remove(path + ".zip")
        self.checkpoints = new_checkpoints
