from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env import ParallelEnv


class ConvertSingleAgent(BaseParallelWrapper):
    def __init__(self, env: ParallelEnv, side: str, opponent_policy=None):
        super().__init__(env)
        assert side in ("player_1", "player_2")
        self.side = side
        self.other_side = "player_1" if side == "player_2" else "player_2"
        self.opponent_policy = opponent_policy
        self._last_opponent_obs = None

    def set_opponent_policy(self, policy):
        self.opponent_policy = policy

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        self._last_opponent_obs = obs[self.other_side]
        return obs[self.side], infos[self.side]

    def step(self, action):
        if self.opponent_policy is not None and self._last_opponent_obs is not None:
            opponent_action = self.opponent_policy(self._last_opponent_obs)
        else:
            opponent_action = self.action_space(self.other_side).sample()

        actions = {
            self.side: action,
            self.other_side: opponent_action,
        }
        obs, rews, terminateds, truncateds, infos = super().step(actions)
        self._last_opponent_obs = obs.get(self.other_side)
        return (
            obs[self.side],
            rews[self.side],
            terminateds[self.side],
            truncateds[self.side],
            infos[self.side],
        )
