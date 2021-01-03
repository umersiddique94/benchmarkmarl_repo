from envs.MultiAgentEnv import MultiAgentEnv
from envs.hns.envs.box_locking import BoxLockingEnv
from envs.hns.envs.blueprint_construction import BlueprintConstructionEnv
from envs.hns.envs.hide_and_seek import HideAndSeekEnv
from gym.spaces import Discrete, Box
import numpy as np
from algorithms.common.common_utils import MultiDiscrete
from functools import reduce

class HNSMultiEnv(MultiAgentEnv):

    def __init__(
            self,
            hns_name,
            num_agents,
            seed
    ):
        if hns_name == "BoxLocking":
            self._env = BoxLockingEnv()
            self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','observation_self']    
            self.mask_order_obs = ['mask_aa_obs','mask_ab_obs','mask_ar_obs',None]            
        elif hns_name == "BlueprintConstruction":
            self._env = BlueprintConstructionEnv()
            self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','construction_site_obs','observation_self']    
            self.mask_order_obs = [None,None,None,None,None]
        elif hns_name == "HideAndSeek":
            self._env = HideAndSeekEnv()
            self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','food_obs','observation_self']    
            self.mask_order_obs = ['mask_aa_obs','mask_ab_obs','mask_ar_obs','mask_af_obs', None]        
        else:
            raise NotImplementedError

        self._env.seed(seed)

        self.num_agents = num_agents
        self.agent_ids = [i for i in range(self.num_agents)]

        all_action_space = []
        all_obs_space = []
        self.have_action_pull = False
        action_movement_dim = []
        
        for agent_id in range(self.num_agents):
            # deal with dict action space
            self.action_movement = self._env.action_space['action_movement'][agent_id].nvec
            action_movement_dim.append(len(self.action_movement))      
            action_glueall = self._env.action_space['action_glueall'][agent_id].n
            action_vec = np.append(self.action_movement, action_glueall)
            if 'action_pull' in self._env.action_space.spaces.keys():
                self.have_action_pull = True
                action_pull = self._env.action_space['action_pull'][agent_id].n
                action_vec = np.append(action_vec, action_pull)
            action_space = MultiDiscrete([[0,vec-1] for vec in action_vec])
            all_action_space.append(action_space) 
            # deal with dict obs space
            obs_space = []
            obs_dim = 0
            for key in self.order_obs:
                if key in self._env.observation_space.spaces.keys():
                    space = list(self._env.observation_space[key].shape)
                    if len(space)<2:  
                        space.insert(0,1)        
                    obs_space.append(space)
                    obs_dim += reduce(lambda x,y:x*y,space)
            obs_space.insert(0,obs_dim)
            all_obs_space.append(obs_space)
        
        self.observation_space = self._convert_to_dict(all_obs_space)
        self.action_space = self._convert_to_dict(all_action_space)

    def reset(self):
        dict_obs = self._env.reset()

        for i, key in enumerate(self.order_obs):
            if key in self._env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:
                    temp_share_obs = dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = dict_obs[key].copy()
                    temp_mask = temp_mask.astype(bool)
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask] = np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs, temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs, temp_share_obs),axis=1)

        obs = self._convert_to_dict(reshape_obs)
        cent_obs = self._convert_to_dict(reshape_share_obs)

        return [obs], [cent_obs]

    def step(self, actions):
        actions = actions[0]
        dict_obs, rew_list, done, info_list  = self._env.step(actions)

        if 'discard_episode' in info_list.keys():
            if info_list['discard_episode']:
                obs = None
                cent_obs = None
            else:
                for i, key in enumerate(self.order_obs):
                    if key in self._env.observation_space.spaces.keys():            
                        if self.mask_order_obs[i] == None:
                            temp_share_obs = dict_obs[key].reshape(self.num_agents,-1).copy()
                            temp_obs = temp_share_obs.copy()
                        else:
                            temp_share_obs = dict_obs[key].reshape(self.num_agents,-1).copy()
                            temp_mask = dict_obs[self.mask_order_obs[i]].copy()
                            temp_obs = dict_obs[key].copy()
                            temp_mask = temp_mask.astype(bool)
                            mins_temp_mask = ~temp_mask
                            temp_obs[mins_temp_mask] = np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                            temp_obs = temp_obs.reshape(self.num_agents,-1) 
                        if i == 0:
                            reshape_obs = temp_obs.copy()
                            reshape_share_obs = temp_share_obs.copy()
                        else:
                            reshape_obs = np.concatenate((reshape_obs, temp_obs),axis=1) 
                            reshape_share_obs = np.concatenate((reshape_share_obs, temp_share_obs),axis=1)

                obs = self._convert_to_dict(reshape_obs)
                cent_obs = self._convert_to_dict(reshape_share_obs)

        done_list = [done] * self.num_agents          
        rewards = self._convert_to_dict(rew_list)
        dones = self._convert_to_dict(done_list)
        
        dones['env'] = all([dones[agent_id] for agent_id in self.agent_ids])
        infos = info_list

        return [obs], [cent_obs], [rewards], [dones], [infos]

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