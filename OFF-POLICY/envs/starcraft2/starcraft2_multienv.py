from envs.MultiAgentEnv import MultiAgentEnv
from envs.starcraft2.StarCraft2 import StarCraft2Env
from envs.vec_env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import time

class StarCraft2MultiEnv(MultiAgentEnv):

    def __init__(
            self,
            map_name="3m",
            step_mul=8,
            move_amount=2,
            difficulty="7",
            game_version=None,
            seed=None,
            continuing_episode=False,
            obs_all_health=True,
            obs_own_health=True,
            obs_last_action=False,
            obs_pathing_grid=False,
            obs_terrain_height=False,
            obs_instead_of_state=False,
            obs_timestep_number=False,
            state_last_action=True,
            state_timestep_number=False,
            reward_sparse=False,
            reward_only_positive=True,
            reward_death_value=10,
            reward_win=200,
            reward_defeat=0,
            reward_negative_scale=0.5,
            reward_scale=True,
            reward_scale_rate=20,
            replay_dir="",
            replay_prefix="",
            window_size_x=1920,
            window_size_y=1200,
            heuristic_ai=False,
            heuristic_rest=False,
            debug=True,
    ):
        self._env = StarCraft2Env(map_name=map_name, step_mul=step_mul, move_amount=move_amount, difficulty=difficulty,
                                  game_version=game_version, seed=seed, continuing_episode=continuing_episode,
                                  obs_all_health=obs_all_health, obs_own_health=obs_own_health,
                                  obs_last_action=obs_last_action,
                                  obs_pathing_grid=obs_pathing_grid, obs_terrain_height=obs_terrain_height,
                                  obs_instead_of_state=obs_instead_of_state,
                                  obs_timestep_number=obs_timestep_number, state_last_action=state_last_action,
                                  state_timestep_number=state_timestep_number,
                                  reward_sparse=reward_sparse, reward_only_positive=reward_only_positive,
                                  reward_death_value=reward_death_value, reward_win=reward_win,
                                  reward_defeat=reward_defeat, reward_negative_scale=reward_negative_scale,
                                  reward_scale=reward_scale, reward_scale_rate=reward_scale_rate,
                                  replay_dir=replay_dir, replay_prefix=replay_prefix, window_size_x=window_size_x,
                                  window_size_y=window_size_y, heuristic_ai=heuristic_ai,
                                  heuristic_rest=heuristic_rest, debug=debug)
        self.num_agents = self._env.n_agents
        self.agent_ids = [i for i in range(self.num_agents)]
        self.num_envs = 1

        self.observation_space = self._convert_to_dict(self._env.observation_space)
        self.share_observation_space = self._convert_to_dict(self._env.share_observation_space)
        self.action_space = self._convert_to_dict(self._env.action_space)

        self.agents = self._env.agents

    def close(self):
        self._env.close()

    def reset(self):
        obs_list, cent_obs_list, available_actions_list = self._env.reset()
        obs = self._convert_to_dict(obs_list)
        cent_obs = {}
        cent_obs['cent_obs'] = cent_obs_list
        available_actions = self._convert_to_dict(available_actions_list)

        return [obs], [cent_obs], [available_actions]

    def step(self, action_dict):
        action_dict = action_dict[0]
        for id in action_dict.keys():
            action = action_dict[id]
            action_space = self.action_space[id]
            converted_action = self._convert_action(action_space, action)
            action_dict[id] = converted_action

        action_list = list(action_dict.values())
        start=time.time()
        obs_list, cent_obs_list, rew_list, done_list, info_list, available_actions_list  = self._env.step(action_list)
        end=time.time()
        print("true step time:")
        print(end-start)
        obs = self._convert_to_dict(obs_list)
        cent_obs = {}
        cent_obs['cent_obs'] = cent_obs_list
        rewards = self._convert_to_dict(rew_list)
        dones = self._form_done_dict(done_list, info_list)
        infos = self._convert_to_dict(info_list)
        available_actions = self._convert_to_dict(available_actions_list)

        return [obs], [cent_obs], [rewards], [dones], [infos], [available_actions]

    def seed(self, seed):
        self._env.seed(seed)

    def render(self):
        self._env.render()

    def _form_done_dict(self, done_list, info_list):
        d = {self.agent_ids[i]: not info_list[i]['high_masks'] for i in range(len(self.agent_ids))}
        d['env'] = all(done_list)
        return d

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
'''
def make_parallel_env(map_name, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = StarCraft2MultiEnv(map_name)
            env.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
'''
def make_parallel_env(map_name, n_rollout_threads, seed):
    env = StarCraft2MultiEnv(map_name)
    env.seed(seed)
    return env