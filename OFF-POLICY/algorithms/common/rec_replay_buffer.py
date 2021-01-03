import numpy as np
import torch
import random


class RecReplayBuffer:
    def __init__(self, max_size, episode_len, policy_ids, agent_ids, policy_agents, policy_obs_dim, policy_cent_obs_dim, policy_act_dim, 
                 use_cent_agent_obs=False, use_available_actions=True):
        self.max_size = max_size
        self.episode_len = episode_len
        self.policy_ids = policy_ids
        self.agent_ids = agent_ids
        self.policy_agents = policy_agents
        self.use_cent_agent_obs = use_cent_agent_obs
        self.use_available_actions = use_available_actions

        self.policy_buffers = {
            p_id: RecPolicyBuffer(p_id, self.max_size, episode_len, self.policy_agents[p_id], policy_obs_dim[p_id], policy_cent_obs_dim[p_id],
                                  policy_act_dim[p_id], self.use_cent_agent_obs, self.use_available_actions) for p_id in self.policy_ids}
        self.size = 0
        

    def push(self, num_envs, obs, cent_obs, acts, rews, nobs, cent_nobs, dones, avail_acts=None, navail_acts=None):
        o, co, a, r, no, cno, d, aa, naa = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for a_id in self.agent_ids:

            o[a_id] = np.stack([np.stack([o_t[i][a_id] for i in range(num_envs)]) for o_t in obs])
            a[a_id] = np.stack([np.stack([a_t[i][a_id] for i in range(num_envs)]) for a_t in acts])
            r[a_id] = np.stack([np.stack([np.array([r_t[i][a_id]]) for i in range(num_envs)]) for r_t in rews])
            no[a_id] = np.stack([np.stack([no_t[i][a_id] for i in range(num_envs)]) for no_t in nobs])
            d[a_id] = np.stack([np.stack([np.array([d_t[i][a_id]]) for i in range(num_envs)]) for d_t in dones])
            if self.use_available_actions:
                aa[a_id] = np.stack([np.stack([aa_t[i][a_id] for i in range(num_envs)]) for aa_t in avail_acts])
                naa[a_id] = np.stack([np.stack([naa_t[i][a_id] for i in range(num_envs)]) for naa_t in navail_acts])
            if self.use_cent_agent_obs:
                co[a_id] = np.stack([np.stack([co_t[i][a_id] for i in range(num_envs)]) for co_t in cent_obs])
                cno[a_id] = np.stack([np.stack([cno_t[i][a_id] for i in range(num_envs)]) for cno_t in cent_nobs])

        d['env'] = np.stack([np.stack([np.array([d_t[i]['env']]) for i in range(num_envs)]) for d_t in dones])
        if not self.use_cent_agent_obs:
            co = np.stack([np.stack([co_t[i]['cent_obs'] for i in range(num_envs)]) for co_t in cent_obs])
            cno = np.stack([np.stack([cno_t[i]['cent_obs'] for i in range(num_envs)]) for cno_t in cent_nobs])

        for p_buffer in self.policy_buffers.values():
            if self.use_available_actions:
                p_buffer.push(num_envs, o, co, a, r, no, cno, d, aa, naa)
            else:
                p_buffer.push(num_envs, o, co, a, r, no, cno, d)

        assert len(set([p_buffer.num_episodes for p_buffer in self.policy_buffers.values()])) == 1
        self.size = self.policy_buffers[self.policy_ids[0]].size

    def sample_chunks(self, batch_size):
        assert self.size > batch_size, "Cannot sample with no completed episodes in the buffer!"

        batch_inds = np.random.choice(self.size, batch_size)
        # batch_inds = np.array([0])
        obs = {}
        cent_obs = {}
        act = {}
        rew = {}
        nobs = {}
        cent_nobs = {}
        dones = {}
        avail_acts = {}
        navail_acts = {}

        done_env_added = False
        for p_id in self.policy_ids:
            p_buffer = self.policy_buffers[p_id]
            o, co, a, r, no, cno, d, d_env, aa, naa = p_buffer.sample_episodes(batch_inds)
            obs[p_id] = o           
            act[p_id] = a
            rew[p_id] = r
            nobs[p_id] = no           
            dones[p_id] = d
            avail_acts[p_id] = aa
            navail_acts[p_id] = naa           
            cent_obs[p_id] = co
            cent_nobs[p_id] = cno

            if not done_env_added:
                dones['env'] = d_env
                done_env_added = True
        return obs, cent_obs, act, rew, nobs, cent_nobs, dones, avail_acts, navail_acts

class RecPolicyBuffer:
    def __init__(self, policy_id, max_size, episode_len, policy_agents, obs_dim, cent_obs_dim, act_dim, use_cent_agent_obs=False, use_available_actions=True):
        self.max_size = max_size
        self.num_agents = len(policy_agents)
        self.policy_id = policy_id
        self.episode_len = episode_len
        self.agent_ids = policy_agents
        self.use_cent_agent_obs = use_cent_agent_obs
        self.use_available_actions = use_available_actions
        
        if isinstance(act_dim, np.ndarray):
            # multidiscrete case
            self.act_dim = int(sum(act_dim))
        else:
            self.act_dim = act_dim

        self.observations = np.zeros((self.num_agents, episode_len, max_size, obs_dim))

        if self.use_cent_agent_obs:
            self.cent_observations = np.zeros((self.num_agents, episode_len, max_size, cent_obs_dim))
        else:
            self.cent_observations = np.zeros((episode_len, max_size, cent_obs_dim))

        self.actions = np.zeros((self.num_agents, episode_len, max_size, self.act_dim))
        self.rewards = np.zeros((self.num_agents, episode_len, max_size,  1))
        self.next_observations = np.zeros_like(self.observations)
        self.next_cent_observations = np.zeros_like(self.cent_observations)
        # default to done being True
        self.dones = np.ones_like(self.rewards).astype(bool)
        self.dones_env = np.ones((episode_len, max_size, 1))
        if self.use_available_actions:
            self.available_actions = np.ones((self.num_agents, episode_len, max_size, self.act_dim))
            self.next_available_actions = np.ones_like(self.available_actions)

        self.num_episodes = 0
        self.num_transitions = 0
        self.size = 0
        self.freeze_size = False


    def push(self, num_eps, obs, cent_obs, acts, rew, nobs, cent_nobs, dones, avail_acts=None, navail_acts=None):
        if self.num_episodes + num_eps >= self.max_size:
            self.size = self.num_episodes
            self.num_episodes = 0
            self.freeze_size = True

        ep_len = None
        for i in range(self.num_agents):
            ep_len = obs[self.agent_ids[i]].shape[0]
            self.observations[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = obs[self.agent_ids[i]]
            self.actions[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = acts[self.agent_ids[i]]
            if self.use_available_actions:
                self.available_actions[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = avail_acts[self.agent_ids[i]]
                self.next_available_actions[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = navail_acts[self.agent_ids[i]]
            self.rewards[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = rew[self.agent_ids[i]]
            self.next_observations[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = nobs[self.agent_ids[i]]
            self.dones[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = dones[self.agent_ids[i]]
            if self.use_cent_agent_obs:
                self.cent_observations[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = cent_obs[self.agent_ids[i]]
                self.next_cent_observations[i, 0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = cent_nobs[self.agent_ids[i]]
        if not self.use_cent_agent_obs:
            self.cent_observations[0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = cent_obs
            self.next_cent_observations[0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = cent_nobs
        self.dones_env[0 : ep_len, self.num_episodes : self.num_episodes + num_eps, :] = dones['env']
        self.num_episodes += num_eps

        if not self.freeze_size:
            self.size = self.num_episodes

    def sample_episodes(self, episode_inds):
        obs = self.observations[:, :, episode_inds, :]
        acts = self.actions[:, :, episode_inds, :]
        rews = self.rewards[:, :, episode_inds, :]
        nobs = self.next_observations[:, :, episode_inds, :]
        dones = self.dones[:, :, episode_inds, :]
        if self.use_cent_agent_obs:
            cent_obs = self.cent_observations[:, :, episode_inds, :]
            cent_nobs = self.next_cent_observations[:, :, episode_inds, :]
        else:
            cent_obs = self.cent_observations[:, episode_inds, :]
            cent_nobs = self.next_cent_observations[:, episode_inds, :]
        dones_env = self.dones_env[:, episode_inds, :]

        if self.use_available_actions:
            avail_acts = self.available_actions[:, :, episode_inds, :]
            navail_acts = self.next_available_actions[:, :, episode_inds, :]
        else:
            avail_acts = None
            navail_acts = None
        
        return obs, cent_obs, acts, rews, nobs, cent_nobs, dones, dones_env, avail_acts, navail_acts
        