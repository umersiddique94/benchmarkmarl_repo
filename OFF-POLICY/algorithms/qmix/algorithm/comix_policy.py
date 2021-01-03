import torch
from algorithms.qmix.algorithm.agent_q_function import AgentQFunction
from algorithms.qmix.utils.epsilon_schedules import DecayThenFlatSchedule
from algorithms.common.common_utils import get_dim_from_space
from gym.spaces import Box
import numpy as np
import tensorflow_probability as tfp


class CoMixPolicy:
    def __init__(self, args, observation_dim, action_space):
        """
        init relevent args
        """
        self.args = args
        self.observation_dim = observation_dim
        self.action_space = action_space
        self.action_dim = get_dim_from_space(action_space)
        self.q_network_input_dim = observation_dim + (2 * self.action_dim)

        self.q_network = AgentQFunction(self.q_network_input_dim, 1, args)

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")

    def get_q_values(self, hidden_batch, observation_batch, prev_action_batch, action_batch=None):
        """
        Get q values for state action pair batch
        Prev_action_batch is onehot row matrix, but action_batch is a nx1 vector (not onehot)
        """
        assert action_batch is not None, "Action batch must be provided for COMIX"
        # combine previous action with observation for input into q
        input_batch = torch.cat([observation_batch.double(), prev_action_batch.double(), action_batch.double()], dim=1).float()
        q_batch, new_hidden_batch = self.q_network(input_batch, hidden_batch)
        # q_batch: matrix of size n x 1

        return q_batch, new_hidden_batch

    def get_actions(self, observation_batch, prev_action_batch, hiddens_batch, eps_greedy, t_env):
        """
        get actions in epsilon-greedy manner, if specified
        """
        batch_size = observation_batch.shape[0]
        greedy_actions = self.cross_entropy_method(observation_batch, prev_action_batch, hiddens_batch)

        # TODO: better sampling strategy? Boltzmann maybe?


        if eps_greedy:
            random_actions = torch.from_numpy(np.vstack([self.action_space.sample() for _ in range(batch_size)]))

            eps = self.schedule.eval(t_env)
            # rand_like samples from Uniform[0, 1]
            rand_numbers = torch.rand_like(observation_batch[:, 0])
            take_random = (rand_numbers < eps).int()
            actions = (1 - take_random) * greedy_actions + take_random * random_actions
        else:
            eps = None
            actions = greedy_actions

        _, new_hiddens = self.get_q_values(hiddens_batch, observation_batch, prev_action_batch, actions)

        return actions.detach(), new_hiddens.detach(), eps

    def make_onehot(self, ind_vector, batch_size):
        onehot_mat = torch.zeros((batch_size, self.action_dim))
        onehot_mat[torch.arange(batch_size), ind_vector.long()] = 1
        return onehot_mat

    def init_hidden(self, batch_size):
        # return {id: self.q_network.init_hidden(batch_size) for id in self.agent_ids}
        return self.q_network.init_hidden(batch_size)

    def parameters(self):
        return self.q_network.parameters()

    def load_state(self, source_policy):
        self.q_network.load_state_dict(source_policy.q_network.state_dict())

    def cross_entropy_method(self, observation_batch, prev_action_batch, hidden_states):
        assert isinstance(self.action_space, Box)

        observation_batch = observation_batch.numpy()
        prev_action_batch = prev_action_batch.numpy()
        hidden_states = hidden_states.numpy()

        batch_size = observation_batch.shape[0]
        mu_batch = np.vstack([self.action_space.sample() for _ in range(batch_size)])
        sig_batch = np.ones_like(mu_batch)

        for i in range(self.args.num_cem_iters):
            num_samples = self.args.num_cem_samples(i)
            dist = tfp.distributions.MultivariateNormalDiag(loc=mu_batch, scale_diag=sig_batch)
            samples = dist.sample(sample_shape=num_samples).numpy().reshape(batch_size, num_samples, self.action_dim)
            samples = np.clip(samples, self.action_space.low, self.action_space.high)

            new_mu_batch = []
            new_sig_batch = []

            if i == self.args.num_cem_iters - 1:
                act_batch = []
            for j in range(batch_size):
                candidate_actions = samples[j]
                related_obs = observation_batch[j].tile(num_samples, 1)
                related_prev_act = prev_action_batch[j].tile(num_samples, 1)
                related_hidden = hidden_states[j].tile(num_samples, 1)
                q_vals, _ = self.get_q_values(related_hidden, related_obs, related_prev_act, candidate_actions)
                cutoff_percentile = self.args.cem_cutoff_percentile
                keep_inds = (q_vals <= np.percentile(q_vals, cutoff_percentile)).reshape(num_samples, )
                kept_actions = candidate_actions[keep_inds, :]

                if i == self.args.num_cem_iters - 1:
                    max_action = candidate_actions[np.argmax(q_vals)]
                    act_batch.append(max_action)

                new_mu = np.mean(kept_actions.astype(np.float64), axis=0)
                new_sig = np.std(kept_actions.astype(np.float64), axis=0)

                new_mu_batch.append(new_mu)
                new_sig_batch.append(new_sig)

            mu_batch = np.vstack(new_mu_batch)
            sig_batch = np.vstack(new_sig_batch)

            if i == self.args.num_cem_iters - 1:
                return torch.from_numpy(np.vstack(act_batch))






