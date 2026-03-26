"""Self-play 상대풀 관리 (PFSP + builtin 포함)."""

import os
import random

import numpy as np
from stable_baselines3 import PPO


def make_opponent_policy(model):
    """SB3 모델을 callable (obs) -> action으로 변환."""
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


def make_builtin_policy():
    """builtin AI용 더미 정책. physics.py가 행동을 덮어쓰므로 아무 값이나 반환."""
    def policy(obs):
        return 0
    return policy


class OpponentPool:
    """PFSP 기반 상대풀 관리.

    상대 선택 비율:
    - latest_prob: 최신 상대 모델
    - builtin_prob: builtin AI
    - 나머지: 풀에서 PFSP 샘플링 (승률 낮은 상대 우선)
    """

    def __init__(self, pool_dir, side, max_pool_size=50):
        self.pool_dir = pool_dir
        self.side = side
        self.max_pool_size = max_pool_size
        self.checkpoints = []
        self.win_stats = {}  # {name: [wins, losses]}
        os.makedirs(pool_dir, exist_ok=True)

    def add_checkpoint(self, model, iteration):
        path = os.path.join(self.pool_dir, f"{self.side}_iter{iteration:06d}")
        model.save(path)
        name = os.path.basename(path)
        self.checkpoints.append(path)
        self.win_stats[name] = [0, 0]
        if len(self.checkpoints) > self.max_pool_size:
            self._prune()
        return path

    def sample_opponent(self, latest_model, latest_prob=0.5, builtin_prob=0.2):
        """상대 선택. latest/builtin/풀(PFSP) 비율로 샘플링."""
        r = random.random()

        if r < latest_prob:
            return latest_model, "latest", False

        if r < latest_prob + builtin_prob:
            return None, "builtin", True  # None 모델, builtin 플래그

        # 풀에서 PFSP 샘플링
        if not self.checkpoints:
            return latest_model, "latest", False

        weights = self._pfsp_weights()
        idx = random.choices(range(len(self.checkpoints)), weights=weights, k=1)[0]
        path = self.checkpoints[idx]
        name = os.path.basename(path)
        model = PPO.load(path, device="cpu")
        return model, name, False

    def update_stats(self, opponent_name, won):
        """상대별 승패 기록 업데이트."""
        if opponent_name not in self.win_stats:
            self.win_stats[opponent_name] = [0, 0]
        if won:
            self.win_stats[opponent_name][0] += 1
        else:
            self.win_stats[opponent_name][1] += 1

    def _pfsp_weights(self):
        """PFSP 가중치: 승률이 낮은 상대에 높은 확률."""
        weights = []
        for path in self.checkpoints:
            name = os.path.basename(path)
            stats = self.win_stats.get(name, [0, 0])
            total = stats[0] + stats[1]
            if total == 0:
                win_rate = 0.5  # 대전 기록 없으면 중립
            else:
                win_rate = stats[0] / total
            # 승률이 낮을수록 높은 가중치
            weights.append(1.0 - win_rate + 0.1)  # +0.1로 최소 확률 보장
        return weights

    def _prune(self):
        """max_pool_size 초과 시 오래된 체크포인트 제거."""
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
                name = os.path.basename(path)
                self.win_stats.pop(name, None)
                if os.path.exists(path + ".zip"):
                    os.remove(path + ".zip")
        self.checkpoints = new_checkpoints
