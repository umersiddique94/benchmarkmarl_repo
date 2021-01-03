from gym.spaces import Discrete, Box
from algorithms.common.common_utils import MultiDiscrete
import numpy as np
from envs.MultiAgentEnv import MultiAgentEnv
from envs.multiagent_particle_envs.mpe.make_env import make_env
from envs.vec_env_wrappers import SubprocVecEnv, DummyVecEnv

class ParticleEnvMultiEnv(MultiAgentEnv):
    def __init__(self, scenario_name):

        self._env = make_env(scenario_name)
        self.num_agents = self._env.n
        self.agent_ids = [i for i in range(self.num_agents)]

        self.observation_space_dict = self._convert_to_dict(self._env.observation_space)
        self.action_space_dict = self._convert_to_dict(self._env.action_space)

        self.agents = self._env.agents

    def reset(self):
        return self._convert_to_dict(self._env.reset())

    def step(self, action_dict):

        for id in action_dict.keys():
            action = action_dict[id]
            action_space = self.action_space_dict[id]
            converted_action = self._convert_action(action_space, action)
            action_dict[id] = converted_action

        action_list = list(action_dict.values())
        obs_list, rew_list, done_list, info_list = self._env.step(action_list)

        obs = self._convert_to_dict(obs_list)
        rewards = self._convert_to_dict(rew_list)
        dones = self._convert_to_dict(done_list)
        dones['env'] = all([dones[agent_id] for agent_id in self.agent_ids])
        infos = self._convert_to_dict([{"done": done} for done in done_list])

        return obs, rewards, dones, infos

    def seed(self, seed):
        self._env.seed(seed)

    def render(self, mode='human'):
        self._env.render(mode=mode)

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

        elif isinstance(action_space, MultiDiscrete):
            if type(action) == list:
                action = np.concatenate(action)
            total_dim = sum((action_space.high - action_space.low) + 1)
            assert type(action) == np.ndarray and len(action) == total_dim, "Invalid MultiDiscrete action!"
            return action
        else:
            raise Exception("Unsupported space")

        return converted_action


def make_parallel_env(scenario_name, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = ParticleEnvMultiEnv(scenario_name)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

