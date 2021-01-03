from envs.MultiAgentEnv import MultiAgentEnv
from envs.hanabi.rl_env import HanabiEnv
from envs.vec_env_wrappers import ShareSubprocVecEnv
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np

class HanabiMultiEnv(MultiAgentEnv):

    def __init__(
            self,
            hanabi_name,
            num_players,
            seed
    ):
        self._env = HanabiEnv(hanabi_name,num_players,seed)
        self.num_agents = self._env.players
        self.agent_ids = [i for i in range(self.num_agents)]

        self.observation_space_dict = self._convert_to_dict(self._env.observation_space)
        self.share_observation_space_dict = self._convert_to_dict(self._env.share_observation_space)
        self.action_space_dict = self._convert_to_dict(self._env.action_space)

        #self.agents = self._env.agents

    def reset(self):
        obs_list, cent_obs_list, available_actions_list = self._env.reset()
        obs = self._convert_to_dict(obs_list)
        cent_obs = self._convert_to_dict(cent_obs_list)
        available_actions = self._convert_to_dict(available_actions_list)

        return obs, cent_obs, available_actions

    def step(self, action_dict):

        for id in action_dict.keys():
            action = action_dict[id]
            action_space = self.action_space_dict[id]
            converted_action = self._convert_action(action_space, action)
            action_dict[id] = converted_action

        action_list = list(action_dict.values())
        obs_list, cent_obs_list, rew_list, done_list, info_list, available_actions_list  = self._env.step(action_list)

        obs = self._convert_to_dict(obs_list)
        cent_obs = self._convert_to_dict(cent_obs_list)
        rewards = self._convert_to_dict(rew_list)
        dones = self._convert_to_dict(done_list)
        
        dones['env'] = all([dones[agent_id] for agent_id in self.agent_ids])
        infos = info_list
        available_actions = self._convert_to_dict(available_actions_list)

        return obs, cent_obs, rewards, dones, infos, available_actions

    def seed(self, seed):
        self._env.seed(seed)

    def render(self):
        self._env.render()

    def _convert_to_dict(self, vals):
        """
        Convert a list of per-agent values into a dict mapping agent_id to the agent's corresponding value.
        Args:
            vals: list of per-agent values. Must be of length self.num_agents
        Returns:
            dict: dictionary mapping agent_id to the agent' corresponding value, as specified in vals
        """
        return dict(zip(self.agent_ids, vals))

    def _convert_action(self, action_space, action):
        if isinstance(action_space, Discrete):
            if type(action) == np.ndarray and len(action) == action_space.n:
                converted_action = action
            else:
                converted_action = np.zeros(action_space.n)
                if type(action) == np.ndarray or type(action) == list:
                    converted_action[action[0]] = 1.0
                else:
                    converted_action[action] = 1.0
        elif isinstance(action_space, Box):
            converted_action = action
        else:
            #TODO(akash): support MultiDiscrete
            raise Exception("Unknown env, must be Discrete or Box")
        return converted_action

def make_parallel_env(hanabi_name, num_players, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = HanabiMultiEnv(hanabi_name, num_players, seed)
            return env
        return init_env
    if n_rollout_threads == 1:
        return ShareSubprocVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])