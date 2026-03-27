"""Self-play 상대풀 관리 (PFSP + builtin 포함)."""

import os
import random
from collections import deque

import numpy as np
from stable_baselines3 import PPO

PFSP_WINDOW = 30  # 슬라이딩 윈도우 크기 (최근 N판만 유지)


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
    - builtin_prob: builtin AI
    - 나머지: 풀에서 PFSP 샘플링 (승률 낮은 상대 우선)

    승률은 슬라이딩 윈도우(최근 30판)로 관리하여 과거 기록이
    현재 에이전트 실력을 반영하지 못하는 문제를 방지.
    """

    def __init__(self, pool_dir, side):
        self.pool_dir = pool_dir
        self.side = side
        self.checkpoints = []
        self.win_stats = {}  # {name: deque([True/False, ...], maxlen=PFSP_WINDOW)}
        os.makedirs(pool_dir, exist_ok=True)

    def add_checkpoint(self, model, iteration):
        path = os.path.join(self.pool_dir, f"{self.side}_iter{iteration:06d}")
        model.save(path)
        name = os.path.basename(path)
        self.checkpoints.append(path)
        self.win_stats[name] = deque(maxlen=PFSP_WINDOW)
        return path

    def sample_opponent(self, latest_model, builtin_prob=0.2):
        """상대 선택. builtin / 풀(PFSP) 비율로 샘플링.

        Args:
            latest_model: 상대방의 현재 학습 중인 모델 (pool 비었을 때 폴백)
            builtin_prob: builtin AI 선택 확률
        """
        r = random.random()

        if r < builtin_prob:
            return None, "builtin", True

        # 풀에서 PFSP 샘플링
        if not self.checkpoints:
            return latest_model, "latest", False  # 풀이 비어있으면 latest 폴백

        weights = self._pfsp_weights()
        idx = random.choices(range(len(self.checkpoints)), weights=weights, k=1)[0]
        path = self.checkpoints[idx]
        name = os.path.basename(path)
        model = PPO.load(path, device="cpu")
        return model, name, False

    def update_stats(self, opponent_name, won):
        """상대별 승패 기록 업데이트 (슬라이딩 윈도우)."""
        if opponent_name not in self.win_stats:
            self.win_stats[opponent_name] = deque(maxlen=PFSP_WINDOW)
        self.win_stats[opponent_name].append(bool(won))

    def get_win_rate(self, opponent_name):
        """특정 상대에 대한 현재 승률 반환."""
        history = self.win_stats.get(opponent_name)
        if not history:
            return 0.5
        return sum(history) / len(history)

    def _pfsp_weights(self):
        """PFSP 가중치: 승률이 낮은 상대에 높은 확률."""
        weights = []
        for path in self.checkpoints:
            name = os.path.basename(path)
            win_rate = self.get_win_rate(name)
            # 승률이 낮을수록 높은 가중치
            weights.append(1.0 - win_rate + 0.1)  # +0.1로 최소 확률 보장
        return weights
