"""ELO 레이팅 계산 및 대전 유틸리티."""

from itertools import combinations

import numpy as np
from stable_baselines3 import PPO

from pikazoo.env.pikazoo_env import raw_env
from pikazoo.wrappers import NormalizeObservation, SimplifyAction

INITIAL_ELO = 1500
K_FACTOR = 32


def update_elo(ra, rb, result, k=K_FACTOR):
    """ELO 레이팅 업데이트. result: 1=A승, 0=B승, 0.5=무승부."""
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    ra_new = ra + k * (result - ea)
    rb_new = rb + k * ((1 - result) - eb)
    return ra_new, rb_new


class Player:
    """대전에 참여하는 플레이어."""

    def __init__(self, name, player_type, model_path=None, model=None):
        self.name = name
        self.player_type = player_type  # "random", "builtin", "model"
        self.model = model
        if player_type == "model" and model_path and model is None:
            self.model = PPO.load(model_path, device="cpu")

    def get_action(self, obs, env, agent_id):
        if self.player_type == "random":
            return env.action_space(agent_id).sample()
        elif self.player_type == "builtin":
            # physics.py가 덮어쓰므로 아무 값이나 반환
            return 0
        elif self.player_type == "model":
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)
        raise ValueError(f"Unknown player type: {self.player_type}")


def make_player(spec):
    """문자열 스펙에서 Player 생성. 'random', 'builtin', 또는 모델 경로."""
    if spec == "random":
        return Player("random", "random")
    elif spec == "builtin":
        return Player("builtin", "builtin")
    else:
        # 모델 경로에서 이름 추출
        name = spec.rstrip("/").split("/")[-1]
        return Player(name, "model", model_path=spec)


def play_game(p1, p2, winning_score=15, seed=None):
    """두 플레이어로 1판 수행. 반환: 1=p1승, 0=p2승."""
    is_p2_computer = p2.player_type == "builtin"
    is_p1_computer = p1.player_type == "builtin"

    env = raw_env(
        winning_score=winning_score,
        serve="winner",
        is_player1_computer=is_p1_computer,
        is_player2_computer=is_p2_computer,
    )
    env = SimplifyAction(env)
    env = NormalizeObservation(env)

    obs, info = env.reset(seed=seed)

    total_rewards = {"player_1": 0.0, "player_2": 0.0}
    while env.agents:
        actions = {
            "player_1": p1.get_action(obs.get("player_1"), env, "player_1"),
            "player_2": p2.get_action(obs.get("player_2"), env, "player_2"),
        }
        obs, rewards, terminated, truncated, infos = env.step(actions)
        for agent in rewards:
            total_rewards[agent] += rewards[agent]

    env.close()
    return 1 if total_rewards["player_1"] > total_rewards["player_2"] else 0


def round_robin(players, games_per_pair=100, winning_score=15, seed=None):
    """라운드 로빈 대전 수행. 결과와 ELO 반환."""
    rng = np.random.default_rng(seed)
    results = {}
    elos = {p.name: INITIAL_ELO for p in players}

    for p1, p2 in combinations(players, 2):
        key = (p1.name, p2.name)
        wins = 0
        for i in range(games_per_pair):
            game_seed = int(rng.integers(0, 2**31))
            result = play_game(p1, p2, winning_score=winning_score, seed=game_seed)
            wins += result
            elos[p1.name], elos[p2.name] = update_elo(
                elos[p1.name], elos[p2.name], result
            )
        results[key] = (wins, games_per_pair - wins)

    return results, elos


def evaluate_model(model_path, opponents=("random", "builtin"), games=100,
                   winning_score=15, seed=None):
    """단일 모델을 여러 상대와 평가. 학습 중 콜백에서 사용."""
    model_player = make_player(model_path)
    rng = np.random.default_rng(seed)
    elos = {model_player.name: INITIAL_ELO}

    results = {}
    for opp_spec in opponents:
        opp = make_player(opp_spec)
        elos.setdefault(opp.name, INITIAL_ELO)

        wins = 0
        for _ in range(games):
            game_seed = int(rng.integers(0, 2**31))
            result = play_game(model_player, opp, winning_score=winning_score,
                               seed=game_seed)
            wins += result
            elos[model_player.name], elos[opp.name] = update_elo(
                elos[model_player.name], elos[opp.name], result
            )
        results[opp.name] = (wins, games - wins)

    return results, elos[model_player.name]
