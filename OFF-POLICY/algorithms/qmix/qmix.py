import torch
import copy
from algorithms.qmix.algorithm.q_mixer import QMixer
from torch.optim.rmsprop import RMSprop
from algorithms.common.common_utils import make_onehot, soft_update
import numpy as np

class QMix:
    def __init__(self, args, agent_ids, policies, policy_mapping_fn, logger, device, episode_length=None):
        """Class to do gradient updates"""
        self.args = args
        self.device = device

        self.lr = self.args.lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        if episode_length is None:
            self.episode_length = self.args.episode_length
        else:
            self.episode_length = episode_length

        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn

        self.agent_ids = sorted(agent_ids)
        self.policy_ids = sorted(list(self.policies.keys()))

        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) * len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # mixer network
        self.mixer = QMixer(args, self.device, multidiscrete_list=multidiscrete_list)
        # target policies/networks
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}
        self.target_mixer = copy.deepcopy(self.mixer)

        # collect all trainable parameters: each policy parameters, and the mixer parameters
        self.parameters = []
        for policy in self.policies.values():
            self.parameters += policy.parameters()
        self.parameters += self.mixer.parameters()
        # TODO: Use Adam
        self.optimizer = torch.optim.Adam(params=self.parameters, lr=self.lr, eps=self.opti_eps)

        # last episode in which target was updated
        self.last_target_update_episode = 0

        if args.double_q:
            print("Double Q Learning will be used")

        self.logger = logger

    def train_on_batch(self, batch, use_cent_agent_obs):
        # unpack the batch
        obs_batch, cent_obs_batch, act_batch, rew_batch, nobs_batch, cent_nobs_batch, dones_batch, avail_act_batch, navail_act_batch = batch

        if use_cent_agent_obs:
            agent_0_pol = self.policy_mapping_fn(self.agent_ids[0])
            cent_obs_batch = torch.FloatTensor(cent_obs_batch[agent_0_pol][0]).to(self.device)
            cent_nobs_batch = torch.FloatTensor(cent_nobs_batch[agent_0_pol][0]).to(self.device)
        else:
            cent_obs_batch = torch.FloatTensor(cent_obs_batch[self.policy_ids[0]]).to(self.device)
            cent_nobs_batch = torch.FloatTensor(cent_nobs_batch[self.policy_ids[0]]).to(self.device)

        rew_batch = torch.FloatTensor(rew_batch[self.policy_ids[0]]).to(self.device)
        dones_batch = torch.FloatTensor(dones_batch['env']).to(self.device)

        # individual agent q value sequences: each element is of shape (ep_len, batch_size, 1)
        agent_q_sequences = []
        # individual agent next step q value sequences
        agent_next_q_sequences = []
        batch_size = None

        for p_id in self.policy_ids:
            # get data related to the policy id
            curr_obs_batch = torch.FloatTensor(obs_batch[p_id]).to(self.device)
            curr_act_batch = torch.FloatTensor(act_batch[p_id]).to(self.device)
            curr_nobs_batch = torch.FloatTensor(nobs_batch[p_id]).to(self.device)
        
            # stack over agents to process them all at once
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2)
            stacked_obs_batch = torch.cat(list(curr_obs_batch), dim=-2)
            stacked_nobs_batch = torch.cat(list(curr_nobs_batch), dim=-2)

            if navail_act_batch[p_id] is not None:
                curr_navail_act_batch = torch.FloatTensor(navail_act_batch[p_id]).to(self.device)
                stacked_navail_act_batch = torch.cat(list(curr_navail_act_batch), dim=-2)
            else:
                stacked_navail_act_batch = None
            

            policy = self.policies[p_id]
            batch_size = curr_obs_batch.shape[2]
            total_batch_size = batch_size * len(self.policy_agents[p_id])
            seq_len = curr_obs_batch.shape[1]
            # form previous action sequence and get all q values for every possible action
            if isinstance(policy.act_dim, np.ndarray):
                # multidiscrete case
                sum_act_dim = int(sum(policy.act_dim))
            else:
                sum_act_dim = policy.act_dim
            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, total_batch_size, sum_act_dim).to(self.device),
                                                 stacked_act_batch[:-1]))
            pol_all_q_out_sequence, pol_final_hidden = policy.get_q_values(stacked_obs_batch, pol_prev_act_buffer_seq, policy.init_hidden(-1, total_batch_size))

            if isinstance(pol_all_q_out_sequence, list):
                # multidiscrete case
                ind = 0
                Q_per_part = []
                for i in range(len(policy.act_dim)):
                    curr_stacked_act_batch = stacked_act_batch[:, :, ind : ind + policy.act_dim[i]]
                    curr_stacked_act_batch_ind = curr_stacked_act_batch.max(dim=-1)[1]
                    curr_all_q_out_sequence = pol_all_q_out_sequence[i]
                    curr_pol_q_out_sequence = torch.gather(curr_all_q_out_sequence, 2, curr_stacked_act_batch_ind.unsqueeze(dim=-1))
                    Q_per_part.append(curr_pol_q_out_sequence)
                    ind += policy.act_dim[i]
                Q_sequence_combined_parts = torch.cat(Q_per_part, dim=-1)
                pol_agents_q_out_sequence = Q_sequence_combined_parts.split(split_size=batch_size, dim=-2)
            else:
                # get the q values associated with the action taken acording ot the batch
                stacked_act_batch_ind = stacked_act_batch.max(dim=-1)[1]
                pol_q_out_sequence = torch.gather(pol_all_q_out_sequence, 2, stacked_act_batch_ind.unsqueeze(dim=-1))
                # separate into agent q sequences for each agent, then cat along the final dimension to prepare for mixer input
                pol_agents_q_out_sequence = pol_q_out_sequence.split(split_size=batch_size, dim=-2)
            agent_q_sequences.append(torch.cat(pol_agents_q_out_sequence, dim=-1))

            target_policy = self.target_policies[p_id]
            with torch.no_grad():
                if isinstance(target_policy.act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(target_policy.act_dim))
                else:
                    sum_act_dim = target_policy.act_dim
                _, new_target_hiddens = target_policy.get_q_values(stacked_obs_batch[0], torch.zeros(total_batch_size, sum_act_dim).to(self.device), target_policy.init_hidden(-1, total_batch_size))

                if self.args.double_q:
                    # actions come from live q; get the q values for the final nobs
                    pol_final_qs, _ = policy.get_q_values(stacked_nobs_batch[-1], stacked_act_batch[-1], pol_final_hidden)

                    if type(pol_final_qs) == list:
                        # multidiscrete case
                        assert stacked_navail_act_batch is None, "Available actions not supported for multidiscrete"
                        pol_nacts = []
                        for i in range(len(pol_final_qs)):
                            pol_final_curr_qs = pol_final_qs[i]
                            pol_all_curr_q_out_seq = pol_all_q_out_sequence[i]
                            pol_all_nq_out_curr_seq = torch.cat((pol_all_curr_q_out_seq[1:], pol_final_curr_qs[None]))
                            pol_curr_nacts = pol_all_nq_out_curr_seq.max(dim=-1)[1]
                            curr_act_dim = policy.act_dim[i]
                            pol_curr_nacts = make_onehot(pol_curr_nacts, total_batch_size, curr_act_dim, seq_len=seq_len)
                            pol_nacts.append(pol_curr_nacts)
                        pol_nacts = torch.cat(pol_nacts, dim=-1)
                        targ_pol_nq_seq, _ = target_policy.get_q_values(stacked_nobs_batch, stacked_act_batch, new_target_hiddens, action_batch=pol_nacts)
                    else:
                        # cat to form all the next step qs
                        pol_all_nq_out_sequence = torch.cat((pol_all_q_out_sequence[1:], pol_final_qs[None]))
                        # mask out the unavailable actions
                        if stacked_navail_act_batch is not None:
                            pol_all_nq_out_sequence[stacked_navail_act_batch == 0.0] = -1e10
                        # greedily choose actions which maximize the q values and convert these actions to onehot
                        pol_nacts = pol_all_nq_out_sequence.max(dim=-1)[1]
                        if isinstance(policy.act_dim, np.ndarray):
                            # multidiscrete case
                            sum_act_dim = int(sum(policy.act_dim))
                        else:
                            sum_act_dim = policy.act_dim
                        pol_nacts = make_onehot(pol_nacts, total_batch_size, sum_act_dim, seq_len=seq_len)

                        # q values given by target but evaluated at actions taken by live
                        targ_pol_nq_seq, _ = target_policy.get_q_values(stacked_nobs_batch, stacked_act_batch, new_target_hiddens, action_batch=pol_nacts)

                else:
                    # just choose actions from target policy
                    _, targ_pol_nq_seq, _, _ = target_policy.get_actions(stacked_nobs_batch, stacked_act_batch, new_target_hiddens, t_env=None, available_actions=stacked_navail_act_batch, explore=False)

                # separate the next qs into sequences for each agent
                pol_agents_nq_sequence = targ_pol_nq_seq.split(split_size=batch_size, dim=-2)
            # cat target qs along the final dim
            agent_next_q_sequences.append(torch.cat(pol_agents_nq_sequence, dim=-1))

        # combine the agent q value sequences to feed into mixer networks
        agent_q_sequences = torch.cat(agent_q_sequences, dim=-1)
        agent_next_q_sequences = torch.cat(agent_next_q_sequences, dim=-1)

        # store the sequences of predicted and next step Q_tot values to form Bellman errors
        predicted_Q_tot_vals = []
        next_step_Q_tot_vals = []
        for t in range(len(agent_q_sequences)):
            curr_state = cent_obs_batch[t]  # global state should be same across agents
            next_state = cent_nobs_batch[t]
            curr_agent_qs = agent_q_sequences[t]
            next_step_agent_qs = agent_next_q_sequences[t]

            curr_Q_tot = self.mixer(curr_agent_qs, curr_state)
            next_step_Q_tot = self.target_mixer(next_step_agent_qs, next_state)

            predicted_Q_tot_vals.append(curr_Q_tot.squeeze(-1))

            next_step_Q_tot_vals.append(next_step_Q_tot.squeeze(-1))

        # stack over time dimension
        predicted_Q_tot_vals = torch.stack(predicted_Q_tot_vals)
        next_step_Q_tot_vals = torch.stack(next_step_Q_tot_vals)
        # all agents must share reward, so get the reward sequence for an agent
        rewards = rew_batch[0]
        # get the done sequence for the env
        dones = dones_batch
        # form bootstrapped targets
        Q_tot_targets = rewards + (1 - dones.float()) * self.args.gamma * next_step_Q_tot_vals
        # form mask to mask out sequence elements corresponding to states at which the episode already ended
        curr_dones_mask = torch.cat(
            (torch.zeros(1, batch_size, 1).float().to(self.device), dones[:self.episode_length - 1, :, :]))

        predicted_Q_tots = predicted_Q_tot_vals * (1 - curr_dones_mask)

        Q_tot_targets = Q_tot_targets * (1 - curr_dones_mask)
        # loss is MSE Bellman Error
        loss = (((predicted_Q_tots - Q_tot_targets.detach()) ** 2).sum()) / (1 - curr_dones_mask).sum()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        return loss, grad_norm, predicted_Q_tots.mean()

    def update_targets(self):
        print("Updating targets")
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(self.policies[policy_id])
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_update_targets(self):
        for policy_id in self.policy_ids:
            soft_update(self.target_policies[policy_id], self.policies[policy_id], self.tau)
        if self.mixer is not None:
            soft_update(self.target_mixer, self.mixer, self.tau)

    def prep_training(self):
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.train()
            self.target_policies[p_id].q_network.train()
        self.mixer.train()
        self.target_mixer.train()

    def prep_rollout(self):
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.eval()
            self.target_policies[p_id].q_network.eval()
        self.mixer.eval()
        self.target_mixer.eval()
