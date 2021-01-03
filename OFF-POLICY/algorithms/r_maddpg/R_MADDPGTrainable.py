import torch
import numpy as np
from algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy
from algorithms.r_maddpg.r_maddpg import R_MADDPG
from algorithms.common.rec_replay_buffer import RecReplayBuffer
from tensorboardX import SummaryWriter
import os

class RMADDPGTrainable(object):

    def __init__(self, config):
        # non-tunable hyperparameters are in args
        self.args = config["args"]
        self.run_dir = config["run_dir"]

        self.batch_size = self.args.batch_size
        self.train_interval_episode = self.args.train_interval_episode
        self.train_interval = self.args.train_interval
        self.test_interval = self.args.test_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.grad_norm_clip = self.args.grad_norm_clip
        self.buffer_size = self.args.buffer_size

        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
      
        self.policy_mapping_fn = config["policy_mapping_fn"]
        self.agent_ids = config["agent_ids"]
        
        self.env = config["env"]
        self.test_env = config["test_env"]       

        # initialize all the policies and organize the agents corresponding to each policy
        self.policies = {p_id: R_MADDPGPolicy(config, self.policy_info[p_id]) for p_id in self.policy_ids}
        if self.args.model_dir is not None:
            self.restore(self.args.model_dir)

        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        self.log_dir = str(self.run_dir / 'logs')       
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger = SummaryWriter(self.log_dir)

        # set tunable hyperparameters
        if config.__contains__("use_parallel_envs"):
            self.num_envs = self.env.num_envs
        else:
            self.num_envs = 1

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_cent_agent_obs"):
            self.use_cent_agent_obs = config["use_cent_agent_obs"]
        else:
            self.use_cent_agent_obs = False

        if config.__contains__("use_available_actions"):
            self.use_available_actions = config["use_available_actions"]
        else:
            self.use_available_actions = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            self.chunk_len = config["buffer_length"]
        else:
            self.episode_length = self.args.episode_length
            self.chunk_len = self.args.episode_length

        self.buffer = RecReplayBuffer(self.buffer_size, self.episode_length, 
                                      self.policy_ids, self.agent_ids,
                                      self.policy_agents, 
                                      self.policy_obs_dim, 
                                      self.policy_central_obs_dim, 
                                      self.policy_act_dim, 
                                      self.use_cent_agent_obs,
                                      self.use_available_actions)

        # initialize rmaddpg class for updating policies
        self.trainer = R_MADDPG(self.args, self.env, self.policies, self.policy_mapping_fn, self.logger, 
                                self.episode_length)

        self.total_env_steps = 0 # total environment interactions collected during training
        self.num_episodes_collected = 0 # total episodes collected during training
        self.total_train_steps = 0 # number of gradient updates performed
        self.last_train_episode = 0 # last episode after which a gradient update was performed
        self.last_train_T = 0
        self.last_test_T = 0 # last episode after which a test run was conducted
        self.last_save_T = 0 # last epsiode after which the models were saved
        self.last_log_T = 0
        self.last_hard_update_episode = 0

        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        # fill replay buffer with random actions
        self.warmup(num_warmup_episodes)

    def train(self):
        train_episode_rewards = []
        train_scores = []
        train_episodes = []
        train_successes = []
        while True:           
            self.logger.add_scalar("collected_episodes", self.num_episodes_collected, global_step=self.total_env_steps)
            self.trainer.prep_rollout()
            if self.use_available_actions:
                if self.take_turn:#hanabi
                    avg_train_rew, train_score = self.collect_rollout_turn(explore=True, training_episode=True, warmup=False)
                    train_scores.append(train_score)
                else:#sc
                    avg_train_rew, _ = self.collect_rollout_avail(explore=True, training_episode=True, warmup=False)
            else:
                if self.use_cent_agent_obs: # hide and seek
                    avg_train_rew, success, discard_episode = self.collect_rollout_cent(explore=True, training_episode=True, warmup=False)
                    train_episodes.append(self.num_envs-discard_episode)
                    train_successes.append(success)
                else: # mpe
                    avg_train_rew = self.collect_rollout(explore=True, training_episode=True, warmup=False)
                    train_episode_rewards.append(avg_train_rew)
            
            # do a gradient update if the number of episodes collected since the last training update exceeds the specified amount
            if self.num_envs > 1:
                pass
            else:
                if ((self.num_episodes_collected - self.last_train_episode) / self.train_interval_episode) >= 1 or self.last_train_episode == 0:
                    # gradient updates
                    self.trainer.prep_training()
                    for p_id in self.policy_ids:
                        sample = self.buffer.sample_chunks(self.batch_size)
                        if self.use_cent_agent_obs:
                            stats = self.trainer.cent_train_policy_on_batch(p_id, sample)
                        else:
                            stats = self.trainer.train_policy_on_batch(p_id, sample)
                        if (self.total_env_steps - self.last_log_T) / self.log_interval >= 1:
                            self.log_stats(p_id, stats, self.total_env_steps)
                            self.last_log_T = self.total_env_steps
                        
                    # polyak update the targets
                    if self.use_soft_update:
                        for pid in self.policy_ids:
                            self.policies[pid].soft_target_updates()
                    else:
                        if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                            for pid in self.policy_ids:
                                self.policies[pid].hard_target_updates()
                            self.last_hard_update_episode = self.num_episodes_collected
                    
                    self.total_train_steps += 1
                    self.last_train_episode = self.num_episodes_collected

            if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
                self.save()
                self.last_save_T = self.total_env_steps

            # collect test episodes if the number of episodes collected since last test run exceeds the specified amount
            if ((self.total_env_steps - self.last_test_T) / self.test_interval) >= 1:
                if self.use_available_actions:
                    if self.take_turn:
                        avg_scores = np.mean(train_scores)
                        print("average_scores is " + str(avg_scores))
                        self.logger.add_scalars("average_scores", {'average_scores': avg_scores}, self.total_env_steps) 
                    else:
                        self.trainer.prep_rollout()
                        avg_test_rew, eval_win_rate = self.collect_test_rollouts()
                        self.logger.add_scalars("test_average_episode_rewards", {'test_average_episode_rewards':avg_test_rew}, self.total_env_steps)
                        self.logger.add_scalars("eval_win_rate", {'eval_win_rate':eval_win_rate}, self.total_env_steps) 
                else:
                    if self.use_cent_agent_obs:
                        success_rate = np.sum(train_successes)/np.sum(train_episodes)
                        print("success rate is " + str(success_rate))
                        self.logger.add_scalars("success_rate", {'success_rate': success_rate}, self.total_env_steps) 
                    else:
                        avg_episode_rewards = np.mean(train_episode_rewards)
                        self.logger.add_scalars("average_episode_rewards", {'test_average_episode_rewards': avg_episode_rewards}, self.total_env_steps)
                        print("average episode rewards is " + str(avg_episode_rewards))
                    
                self.last_test_T = self.total_env_steps
                break
          
    def save(self):
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(), critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(), actor_save_path + '/actor.pt')

    def restore(self, checkpoint):
        for pid in self.policy_ids:
            path = checkpoint + str(pid)
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)
    # hanabi
    def collect_rollout_turn(self, explore=True, training_episode=True, warmup=False):      
        env = self.env if training_episode or warmup else self.test_env
        success_to_collect_one_episode = False
        while not success_to_collect_one_episode:
            scores = 0
            ep_rewards = [0 for _ in range(self.num_envs)]
            # init RNN states
            actor_rnn_states = {a_id: self.policies[self.policy_mapping_fn(a_id)].init_hidden(-1, self.num_envs, use_numpy=True) for a_id in
                                self.agent_ids}

            agent_prev_actions = {a_id: np.zeros((self.num_envs, self.policy_act_dim[self.policy_mapping_fn(a_id)])) for
                                  a_id in self.agent_ids}
                                
            ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones, ep_avail_acts, ep_navail_acts = [], [], [], [], [], [], [], [], []

            obs, cent_obs, available_actions = env.reset()
            terminate_episodes = False
            t = 0
            turn_count = 0
            turn_rew_since_last_action = np.zeros((self.num_envs, len(self.agent_ids))).astype(np.float32)
            env_actions = [dict() for _ in range(self.num_envs)]
            turn_acts = [dict() for _ in range(self.num_envs)]
            turn_obs = [dict() for _ in range(self.num_envs)]
            turn_cent_obs = [dict() for _ in range(self.num_envs)]
            turn_avail_acts = [dict() for _ in range(self.num_envs)]
            turn_rew = [dict() for _ in range(self.num_envs)]
            turn_dones = [dict() for _ in range(self.num_envs)]

            turn_acts_last = [dict() for _ in range(self.num_envs)]
            turn_obs_last = [dict() for _ in range(self.num_envs)]
            turn_nobs_last = [dict() for _ in range(self.num_envs)]
            turn_cent_obs_last = [dict() for _ in range(self.num_envs)]
            turn_cent_nobs_last = [dict() for _ in range(self.num_envs)]
            turn_avail_acts_last = [dict() for _ in range(self.num_envs)]
            turn_navail_acts_last = [dict() for _ in range(self.num_envs)]
            turn_rew_last = [dict() for _ in range(self.num_envs)]
            turn_dones_last = [dict() for _ in range(self.num_envs)]
    
            while t < self.episode_length:
                # get actions for all agents to step the env 
                env_t = 0          
                for agent_id in self.agent_ids:
                    policy = self.policies[self.policy_mapping_fn(agent_id)]
                
                    obs_batch = np.vstack([o[agent_id] for o in obs])
                    cent_obs_batch = np.vstack([co[agent_id] for co in cent_obs])
                    available_actions_batch = np.vstack([ac[agent_id] for ac in available_actions])
                    if warmup:
                        # completely random actions in pre-training warmup phase
                        act_batch = policy.get_random_actions(obs_batch, available_actions_batch)
                        # get new rnn hidden state
                        _, new_actor_rnn_states, _ = policy.get_actions(obs_batch, agent_prev_actions[agent_id],
                                                                        actor_rnn_states[agent_id], available_actions_batch)
                        eps = None
                    else:
                        # get actions with exploration noise (eps-greedy/Gaussian)
                        act_batch, new_actor_rnn_states, eps = policy.get_actions(obs_batch,
                                                                                agent_prev_actions[agent_id],
                                                                                actor_rnn_states[agent_id],
                                                                                available_actions_batch,
                                                                                t_env=self.total_env_steps,
                                                                                use_target=False, use_gumbel=False,
                                                                                explore=explore)
                    if not isinstance(act_batch, np.ndarray):
                        act_batch = act_batch.detach().numpy()

                    # update rnn hidden state
                    new_actor_rnn_states = new_actor_rnn_states.detach().numpy()
                    actor_rnn_states[agent_id] = new_actor_rnn_states
                    agent_prev_actions[agent_id] = act_batch
                    if eps is not None:
                        self.logger.add_scalar("exploration_eps", eps, global_step=self.total_env_steps)

                    # unpack actions to format needed to step env (list of dicts, dict mapping agent_id to action)                              
                    for i in range(self.num_envs):
                        env_actions[i][agent_id] = act_batch[i]
                        turn_acts[i][agent_id] = act_batch[i]
                        turn_obs[i][agent_id] = obs_batch[i]
                        turn_cent_obs[i][agent_id] = cent_obs_batch[i]
                        turn_avail_acts[i][agent_id] = available_actions_batch[i]

                    if turn_count == 0:
                        pass
                    else:
                        for i in range(self.num_envs):
                            turn_acts_last[i][agent_id] = turn_acts[i][agent_id]
                            turn_obs_last[i][agent_id] = turn_obs[i][agent_id]
                            turn_cent_obs_last[i][agent_id] = turn_cent_obs[i][agent_id]
                            turn_avail_acts_last[i][agent_id] = turn_avail_acts[i][agent_id] 
                            turn_rew_last[i][agent_id] = turn_rew[i][agent_id]
                            turn_dones_last[i][agent_id] = turn_dones[i][agent_id]
                            turn_dones_last[i]["env"] = turn_dones[i]["env"]                   
                            turn_nobs_last[i][agent_id] = obs_batch[i]
                            turn_cent_nobs_last[i][agent_id] = cent_obs_batch[i]
                            turn_navail_acts_last[i][agent_id] = available_actions_batch[i]

                    # env step and store the relevant episode information
                    next_obs, cent_next_obs, rew, done, info, next_available_actions = env.step(env_actions)
                    
                    t += 1
                    env_t += self.num_envs
                    
                    for i in range(self.num_envs):
                        for k in self.agent_ids:
                            turn_rew_since_last_action[i][k] += rew[i][k]
                        turn_rew[i][agent_id] = turn_rew_since_last_action[i][agent_id]
                        turn_rew_since_last_action[i][agent_id] = 0.0
                        turn_dones[i][agent_id] = done[i][agent_id]
                        turn_dones[i]["env"] = done[i]["env"]
                        if (done[i]["env"] or (t == self.episode_length - 1)):
                            # episode is done
                            current_agent_id = agent_id
                            # current_agent has normal values, copy them to buffer
                            turn_acts_last[i][current_agent_id] = turn_acts[i][current_agent_id]
                            turn_obs_last[i][current_agent_id] = turn_obs[i][current_agent_id]
                            turn_cent_obs_last[i][current_agent_id] = turn_cent_obs[i][current_agent_id]
                            turn_avail_acts_last[i][current_agent_id] = turn_avail_acts[i][current_agent_id] 
                            turn_rew_last[i][current_agent_id] = turn_rew[i][current_agent_id] 

                            # any value is okay
                            turn_nobs_last[i][current_agent_id] = np.zeros(self.policies['policy_0'].obs_dim)
                            turn_cent_nobs_last[i][current_agent_id] = np.zeros(self.policies['policy_0'].central_obs_dim)
                            turn_navail_acts_last[i][current_agent_id] = np.ones(self.policies['policy_0'].act_dim) 

                            # deal with left_agent of this turn
                            for left_agent_id in self.agent_ids:
                                if left_agent_id > current_agent_id:
                                    # must use the right value
                                    turn_rew[i][left_agent_id] = turn_rew_since_last_action[i][left_agent_id]
                                    turn_rew_since_last_action[i][left_agent_id] = 0.0
                                    turn_rew_last[i][left_agent_id] = turn_rew[i][left_agent_id]

                                    # use any value is okay
                                    turn_acts_last[i][left_agent_id] = np.zeros(self.policies['policy_0'].act_dim)
                                    turn_obs_last[i][left_agent_id] = np.zeros(self.policies['policy_0'].obs_dim)
                                    turn_cent_obs_last[i][left_agent_id] = np.zeros(self.policies['policy_0'].central_obs_dim)
                                    turn_avail_acts_last[i][left_agent_id] = np.ones(self.policies['policy_0'].act_dim)
                                    turn_nobs_last[i][left_agent_id] = np.zeros(self.policies['policy_0'].obs_dim)
                                    turn_cent_nobs_last[i][left_agent_id] = np.zeros(self.policies['policy_0'].central_obs_dim)
                                    turn_navail_acts_last[i][left_agent_id] = np.ones(self.policies['policy_0'].act_dim)

                            for j in self.agent_ids:
                                turn_dones[i][j] = True

                            turn_dones_last[i] = turn_dones[i]

                            if 'score' in info[i].keys():
                                scores = info[i]['score']

                            terminate_episodes = True
                            break

                    if terminate_episodes:
                        break

                    obs = next_obs
                    cent_obs = cent_next_obs
                    available_actions = next_available_actions
                
                if turn_count > 0:
                    if training_episode or warmup:
                        self.total_env_steps += env_t      
                    ep_obs.append(turn_obs_last)
                    ep_nobs.append(turn_nobs_last)
                    ep_cent_obs.append(turn_cent_obs_last)
                    ep_cent_nobs.append(turn_cent_nobs_last)
                    ep_acts.append(turn_acts_last)
                    ep_rews.append(turn_rew_last)           
                    ep_dones.append(turn_dones_last)
                    ep_avail_acts.append(turn_avail_acts_last)
                    ep_navail_acts.append(turn_navail_acts_last)

                if terminate_episodes:
                    break
                turn_count += 1

            if (training_episode or warmup) and turn_count > 0:
                # push all episodes collected in this rollout step to the buffer 
                success_to_collect_one_episode = True          
                self.buffer.push(self.num_envs, ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones, ep_avail_acts, ep_navail_acts)

            avg_reward = np.mean(np.array(ep_rewards))

            if not warmup and training_episode and turn_count > 0:
                self.num_episodes_collected += self.num_envs

        return avg_reward, scores
    # sc
    def collect_rollout_avail(self, explore=True, training_episode=True, warmup=False):
        battles_won = 0
        env = self.env if training_episode or warmup else self.test_env

        ep_rewards = [0 for _ in range(self.num_envs)]
        # init RNN states
        rnn_states = {
            p_id: self.policies[p_id].init_hidden(-1, self.num_envs * len(self.policy_agents[p_id]), use_numpy=True) for
            p_id in self.policy_ids}
        pol_prev_acts = {p_id: np.zeros(((self.num_envs * len(self.policy_agents[p_id])), self.policy_act_dim[p_id])) for
                         p_id in self.policy_ids}
                              
        ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones, ep_avail_acts, ep_navail_acts = [], [], [], [], [], [], [], [], []

        obs, cent_obs, available_actions = env.reset()
        t = 0
        while t < self.episode_length:
            # get actions for all agents to step the env
            env_actions = [dict() for _ in range(self.num_envs)]

            for p_id in self.policy_ids:
                policy = self.policies[p_id]
               
                obs_batch = np.concatenate([np.stack([o[agent_id] for o in obs]) for agent_id in self.policy_agents[p_id]])
                available_actions_batch = np.concatenate([np.stack([ac[agent_id] for ac in available_actions]) for agent_id in self.policy_agents[p_id]])
                if warmup:
                    # completely random actions in pre-training warmup phase
                    act_batch = policy.get_random_actions(obs_batch, available_actions_batch)
                    # get new rnn hidden state
                    _, new_rnn_states, _ = policy.get_actions(obs_batch, pol_prev_acts[p_id],
                                                                    rnn_states[p_id], available_actions_batch)
                    eps = None
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    act_batch, new_rnn_states, eps = policy.get_actions(obs_batch,
                                                                            pol_prev_acts[p_id],
                                                                            rnn_states[p_id],
                                                                            available_actions_batch,
                                                                            t_env=self.total_env_steps,
                                                                            use_target=False, use_gumbel=False,
                                                                            explore=explore)
                if not isinstance(act_batch, np.ndarray):
                    act_batch = act_batch.detach().numpy()
                # update rnn hidden state
                new_rnn_states = new_rnn_states.detach().numpy()
                rnn_states[p_id] = new_rnn_states
                pol_prev_acts[p_id] = act_batch
                if eps is not None:
                    self.logger.add_scalar("exploration_eps", eps, global_step=self.total_env_steps)

                agent_acts = np.split(act_batch, len(self.policy_agents[p_id]))

                # unpack actions to format needed to step env (list of dicts, dict mapping agent_id to action)
                for i in range(len(self.policy_agents[p_id])):
                    agent_id = self.policy_agents[p_id][i]
                    for j in range(self.num_envs):
                        env_actions[j][agent_id] = agent_acts[i][j]

            # env step and store the relevant episode information
            next_obs, cent_next_obs, rew, done, info, next_available_actions = env.step(env_actions)

            t += 1

            ep_obs.append(obs)
            ep_cent_obs.append(cent_obs)
            ep_acts.append(env_actions)
            ep_rews.append(rew)
            ep_nobs.append(next_obs)
            ep_cent_nobs.append(cent_next_obs)
            ep_dones.append(done)
            ep_avail_acts.append(available_actions)
            ep_navail_acts.append(next_available_actions)

            terminate_episodes = any(
                [d["env"] for d in done]) or t == self.episode_length - 1  # TODO: change any to all?

            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            for i in range(self.num_envs):
                ep_rewards[i] += list(rew[i].values())[0]  # shared reward so take any reward
                # TODO: change to allow for unshared reward

            obs = next_obs
            cent_obs = cent_next_obs
            available_actions = next_available_actions           

            if terminate_episodes:
                for i in range(self.num_envs):
                    if 'won' in info[i][0].keys():
                        if info[i][0]['won']: #take one agent
                            battles_won += 1
                break

        if training_episode or warmup:
            # push all episodes collected in this rollout step to the buffer           
            self.buffer.push(self.num_envs, ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones, ep_avail_acts, ep_navail_acts)

        avg_reward = np.mean(np.array(ep_rewards))

        if not warmup and training_episode:
            self.num_episodes_collected += self.num_envs

        return avg_reward, battles_won
    # hide and seek
    def collect_rollout_cent(self, explore=True, training_episode=True, warmup=False):
        success = 0
        discard_episode = 0
        env = self.env if training_episode or warmup else self.test_env

        ep_rewards = [0 for _ in range(self.num_envs)]
        # init RNN states
        rnn_states = {p_id: self.policies[p_id].init_hidden(-1, self.num_envs * len(self.policy_agents[p_id]), use_numpy=True) for
                        p_id in self.policy_ids}
        
        pol_prev_acts = {}
        for p_id in self.policy_ids:
            temp_act_dim = self.policy_act_dim[p_id]
            if isinstance(temp_act_dim, np.ndarray):
                # multidiscrete case
                self.sum_act_dim = int(sum(temp_act_dim))
            else:
                self.sum_act_dim = temp_act_dim
            pol_prev_acts[p_id] = np.zeros(((self.num_envs * len(self.policy_agents[p_id])), self.sum_act_dim))
                                 
        ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones = [], [], [], [], [], [], []

        obs, cent_obs = env.reset()
        t = 0
        while t < self.episode_length:
            # get actions for all agents to step the env

            buffer_actions = [dict() for _ in range(self.num_envs)]

            for p_id in self.policy_ids:
                policy = self.policies[p_id]
               
                obs_batch = np.concatenate([np.stack([o[agent_id] for o in obs]) for agent_id in self.policy_agents[p_id]])
                if warmup:
                    # completely random actions in pre-training warmup phase
                    act_batch = policy.get_random_actions(obs_batch)
                    # get new rnn hidden state
                    _, new_rnn_states, _ = policy.get_actions(obs_batch, pol_prev_acts[p_id], rnn_states[p_id])
                    eps = None
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    act_batch, new_rnn_states, eps = policy.get_actions(obs_batch,
                                                                            pol_prev_acts[p_id],
                                                                            rnn_states[p_id],
                                                                            t_env=self.total_env_steps,
                                                                            use_target=False, use_gumbel=False,
                                                                            explore=explore)
                if not isinstance(act_batch, np.ndarray):
                    act_batch = act_batch.detach().numpy()
                # update rnn hidden state
                new_rnn_states = new_rnn_states.detach().numpy()
                rnn_states[p_id] = new_rnn_states
                pol_prev_acts[p_id] = act_batch
                if eps is not None:
                    self.logger.add_scalar("exploration_eps", eps, global_step=self.total_env_steps)

                agent_actions = np.split(act_batch, len(self.policy_agents[p_id]))

                # unpack actions to format needed to step env (list of dicts, dict mapping agent_id to action)
                for i in range(len(self.policy_agents[p_id])):
                    agent_id = self.policy_agents[p_id][i]
                    for j in range(self.num_envs):
                        buffer_actions[j][agent_id] = agent_actions[i][j]
                                   
            env_actions = []    
            for i in range(self.num_envs):
                action_movement = []
                action_pull = []
                action_glueall = []
                for agent_id in self.agent_ids:
                    temp_action_movement = np.zeros_like(env.action_movement)
                    for k,movement_dim in enumerate(env.action_movement):
                        temp_action_movement[k] = np.argmax(agent_actions[agent_id][i][k*movement_dim:(k+1)*movement_dim-1])
                    action_movement.append(temp_action_movement)
                    glueall_dim_start = np.sum(env.action_movement)
                    action_glueall.append(int(np.argmax(agent_actions[agent_id][i][glueall_dim_start:glueall_dim_start+2])))
                    
                    if env.have_action_pull:
                        action_pull.append(int(np.argmax(agent_actions[agent_id][i][-2:])))
                action_movement = np.stack(action_movement, axis = 0)
                action_glueall = np.stack(action_glueall, axis = 0)
                if env.have_action_pull:
                    action_pull = np.stack(action_pull, axis = 0)                             
                one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
                env_actions.append(one_env_action)

            # env step and store the relevant episode information
            next_obs, cent_next_obs, rew, done, info = env.step(env_actions)

            t += 1

            for i in range(self.num_envs):
                if 'discard_episode' in info[i].keys():
                    if info[i]['discard_episode']: #take one agent
                        discard_episode += 1
                        next_obs = obs
                        cent_next_obs = cent_obs

            ep_obs.append(obs)
            ep_cent_obs.append(cent_obs)
            ep_acts.append(buffer_actions)
            ep_rews.append(rew)
            ep_nobs.append(next_obs)
            ep_cent_nobs.append(cent_next_obs)
            ep_dones.append(done)

            terminate_episodes = any(
                [d["env"] for d in done]) or t == self.episode_length - 1  # TODO: change any to all?

            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            for i in range(self.num_envs):
                ep_rewards[i] += list(rew[i].values())[0]  # shared reward so take any reward
                # TODO: change to allow for unshared reward

            obs = next_obs
            cent_obs = cent_next_obs           

            if terminate_episodes:
                for i in range(self.num_envs):
                    if 'success' in info[i].keys():
                        if info[i]['success']: #take one agent
                            success += 1
                break

        if training_episode or warmup:
            # push all episodes collected in this rollout step to the buffer           
            self.buffer.push(self.num_envs, ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones)

        avg_reward = np.mean(np.array(ep_rewards))

        if not warmup and training_episode:
            self.num_episodes_collected += self.num_envs

        return avg_reward, success, discard_episode
    # mpe
    def collect_rollout(self, explore=True, training_episode=True, warmup=False):
        
        env = self.env if training_episode or warmup else self.test_env

        ep_rewards = [0 for _ in range(self.num_envs)]
        # init RNN states
        rnn_states = {p_id: self.policies[p_id].init_hidden(-1, self.num_envs * len(self.policy_agents[p_id]), use_numpy=True) for
                        p_id in self.policy_ids}
        
        pol_prev_acts = {}
        for p_id in self.policy_ids:
            temp_act_dim = self.policy_act_dim[p_id]
            if isinstance(temp_act_dim, np.ndarray):
                # multidiscrete case
                self.sum_act_dim = int(sum(temp_act_dim))
            else:
                self.sum_act_dim = temp_act_dim
            pol_prev_acts[p_id] = np.zeros(((self.num_envs * len(self.policy_agents[p_id])), self.sum_act_dim))
                                 
        ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones = [], [], [], [], [], [], []

        obs = env.reset()

        t = 0
        while t < self.episode_length:
            # get actions for all agents to step the env

            env_actions = [dict() for _ in range(self.num_envs)]

            for p_id in self.policy_ids:
                policy = self.policies[p_id]

                obs_batch = np.concatenate(
                    [np.stack([o[agent_id] for o in obs]) for agent_id in self.policy_agents[p_id]])

                cent_obs = []
                for o in obs:
                    agent_obs = []
                    for agent_id in self.agent_ids:
                        agent_obs.append(o[agent_id])
                    cat_agent_obs = np.concatenate(agent_obs)
                    cent_obs.append({'cent_obs':cat_agent_obs.reshape(-1)})

                if warmup:
                    # completely random actions in pre-training warmup phase
                    act_batch = policy.get_random_actions(obs_batch)
                    # get new rnn hidden state
                    _, new_rnn_states, _ = policy.get_actions(obs_batch, pol_prev_acts[p_id], rnn_states[p_id])
                    eps = None
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    act_batch, new_rnn_states, eps = policy.get_actions(obs_batch,
                                                                            pol_prev_acts[p_id],
                                                                            rnn_states[p_id],
                                                                            t_env=self.total_env_steps,
                                                                            use_target=False, use_gumbel=False,
                                                                            explore=explore)
                if not isinstance(act_batch, np.ndarray):
                    act_batch = act_batch.detach().numpy()
                # update rnn hidden state
                new_rnn_states = new_rnn_states.detach().numpy()
                rnn_states[p_id] = new_rnn_states
                pol_prev_acts[p_id] = act_batch
                if eps is not None:
                    self.logger.add_scalar("exploration_eps", eps, global_step=self.total_env_steps)

                agent_actions = np.split(act_batch, len(self.policy_agents[p_id]))

                # unpack actions to format needed to step env (list of dicts, dict mapping agent_id to action)
                for i in range(len(self.policy_agents[p_id])):
                    agent_id = self.policy_agents[p_id][i]
                    for j in range(self.num_envs):
                        env_actions[j][agent_id] = agent_actions[i][j]

            # env step and store the relevant episode information
            next_obs, rew, done, info = env.step(env_actions)

            cent_next_obs = []
            for no in next_obs:
                next_agent_obs = []
                for agent_id in self.agent_ids:
                    next_agent_obs.append(no[agent_id])
                cat_next_agent_obs = np.concatenate(next_agent_obs)
                cent_next_obs.append({'cent_obs':cat_next_agent_obs.reshape(-1)})

            t += 1

            ep_obs.append(obs)
            ep_cent_obs.append(cent_obs)
            ep_acts.append(env_actions)
            ep_rews.append(rew)
            ep_nobs.append(next_obs)
            ep_cent_nobs.append(cent_next_obs)
            ep_dones.append(done)

            terminate_episodes = any(
                [d["env"] for d in done]) or t == self.episode_length - 1  # TODO: change any to all?

            if training_episode or warmup:
                self.total_env_steps += self.num_envs

            for i in range(self.num_envs):
                ep_rewards[i] += list(rew[i].values())[0]  # shared reward so take any reward
                # TODO: change to allow for unshared reward

            obs = next_obs 
            
            if self.num_envs > 1 and not warmup and training_episode:
                if ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1 or self.last_train_T == 0:
                    # gradient updates
                    self.trainer.prep_training()
                    for p_id in self.policy_ids:
                        sample = self.buffer.sample_chunks(self.batch_size)
                        if self.use_cent_agent_obs:
                            stats = self.trainer.cent_train_policy_on_batch(p_id, sample)
                        else:
                            stats = self.trainer.train_policy_on_batch(p_id, sample)
                        if (self.total_env_steps - self.last_log_T) / self.log_interval >= 1:
                            self.log_stats(p_id, stats, self.total_env_steps)
                            self.last_log_T = self.total_env_steps
                        
                    # polyak update the targets
                    if self.use_soft_update:
                        for pid in self.policy_ids:
                            self.policies[pid].soft_target_updates()
                    else:
                        if ((self.num_episodes_collected - self.last_hard_update_episode) / self.hard_update_interval_episode) >= 1:
                            for pid in self.policy_ids:
                                self.policies[pid].hard_target_updates()
                            self.last_hard_update_episode = self.num_episodes_collected
                    
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps
            else:
                pass          

            if terminate_episodes:
                break

        if training_episode or warmup:
            # push all episodes collected in this rollout step to the buffer           
            self.buffer.push(self.num_envs, ep_obs, ep_cent_obs, ep_acts, ep_rews, ep_nobs, ep_cent_nobs, ep_dones)

        avg_reward = np.mean(np.array(ep_rewards))

        if not warmup and training_episode:
            self.num_episodes_collected += self.num_envs

        return avg_reward

    def warmup(self, num_warmup_episodes):
        # fill replay buffer with enough episodes to begin training
        self.trainer.prep_rollout()
        warmup_rew = 0
        num_iters = (num_warmup_episodes // self.num_envs) + 1
        for _ in range(num_iters):
            print("in warm up now")
            if self.use_available_actions:
                if self.take_turn:
                    rew, _ = self.collect_rollout_turn(explore=True, training_episode=False, warmup=True)
                else:
                    rew, _ = self.collect_rollout_avail(explore=True, training_episode=False, warmup=True)
            else:
                if self.use_cent_agent_obs:
                    rew, _, _ = self.collect_rollout_cent(explore=True, training_episode=False, warmup=True)
                else:
                    rew = self.collect_rollout(explore=True, training_episode=False, warmup=True)
            warmup_rew += rew
        warmup_rew = warmup_rew / num_iters
        print("Warmup average episode rewards: ", warmup_rew)

    def collect_test_rollouts(self):
        average_rewards = []
        battles_wons = []
        successes = []
        scores = []
        for _ in range(self.args.num_test_episodes):
            if self.use_available_actions:
                if self.take_turn:
                    average_reward, score = self.collect_rollout_turn(explore=False, training_episode=False, warmup=False)
                    scores.append(score)
                else:
                    average_reward, battles_won = self.collect_rollout_avail(explore=False, training_episode=False, warmup=False)
                    battles_wons.append(battles_won) 
            else:
                if self.use_cent_agent_obs:
                    average_reward, success, _ = self.collect_rollout_cent(explore=False, training_episode=False, warmup=False)
                    successes.append(success) 
                else:
                    average_reward = self.collect_rollout(explore=False, training_episode=False, warmup=False)         
            average_rewards.append(average_reward)
            
        average_rewards = np.mean(np.array(average_rewards))
        if self.use_available_actions:
            if self.take_turn:
                avg_scores = np.mean(np.array(scores))
                print("average score is %.6f" % avg_scores)
                return average_rewards, avg_scores
            else:
                battles_won_rate = np.sum(np.array(battles_wons))/self.args.num_test_episodes
                print("eval win rate is %.6f" % battles_won_rate)
                return average_rewards, battles_won_rate
        else:
            if self.use_cent_agent_obs:
                success_rate = np.mean(np.array(successes))
                print("success rate is %.6f" % success_rate)
                return average_rewards, success_rate
            else:
                print("test average episode reward is %.6f" % np.mean(average_rewards))
                return average_rewards
        
    def log_stats(self, policy_id, stats, t_env):
        # unpack the statistics
        critic_loss, actor_loss, critic_grad_norm, actor_grad_norm = stats
        # log into tensorboard
        self.logger.add_scalar(str(policy_id) + '/critic_loss', critic_loss, global_step=t_env)
        self.logger.add_scalar(str(policy_id) + '/actor_loss', actor_loss, global_step=t_env)
        self.logger.add_scalar(str(policy_id) + '/critic_grad_norm', critic_grad_norm, global_step=t_env)
        self.logger.add_scalar(str(policy_id) + '/actor_grad_norm', actor_grad_norm, global_step=t_env)
