import torch
from algorithms.qmix.algorithm.agent_q_function import AgentQFunction
from torch.distributions import Categorical
from algorithms.common.common_utils import get_dim_from_space, make_onehot, DecayThenFlatSchedule, avail_choose, MultiDiscrete


class QMixPolicy:
    def __init__(self, config, policy_config):
        """
        init relevent args
        """
        self.args = config["args"]
        self.device = config['device']
        self.obs_dim = policy_config["obs_dim"]
        self.action_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.action_space)
        self.hidden_size = self.args.hidden_size
        self.central_obs_dim = policy_config["cent_obs_dim"]        
        self.multidiscrete = isinstance(self.action_space, MultiDiscrete)       

        if self.args.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.q_network_input_dim = self.obs_dim + self.act_dim
        else:
            self.q_network_input_dim = self.obs_dim
        # Local recurrent q network for the agent
        self.q_network = AgentQFunction(self.q_network_input_dim, self.act_dim, self.args, self.device)

        self.schedule = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish, self.args.epsilon_anneal_time,
                                              decay="linear")

    def get_q_values(self, observation_batch, prev_action_batch, hidden_states, action_batch=None):
        """
        Get q values for state action pair batch
        Prev_action_batch: batch_size x action_dim, rows are onehot is onehot row matrix, but action_batch is a nx1 vector (not onehot)
        """
        if len(observation_batch.shape) == 3:
            sequence = True
            batch_size = observation_batch.shape[1]
        else:
            sequence = False
            batch_size = observation_batch.shape[0]

        # combine previous action with observation for input into q, if specified in args
        if self.args.prev_act_inp:
            input_batch = torch.cat((observation_batch, prev_action_batch), dim=-1)
        else:
            input_batch = observation_batch.float()
        hidden_states = hidden_states.float()
        q_batch, new_hidden_batch = self.q_network(input_batch, hidden_states)

        if action_batch is not None:
            if self.multidiscrete:
                ind = 0
                all_q_values = []
                for i in range(len(self.act_dim)):
                    curr_q_batch = q_batch[i]
                    curr_action_portion = action_batch[:, :, ind : ind + self.act_dim[i]]
                    curr_action_inds = curr_action_portion.max(dim=-1)[1].to(self.device)
                    curr_q_values = torch.gather(curr_q_batch, 2, curr_action_inds.unsqueeze(dim=-1))
                    all_q_values.append(curr_q_values)
                    ind += self.act_dim[i]
                return torch.cat(all_q_values, dim=-1), new_hidden_batch
            else:
                # convert one-hot action batch to index tensors to gather the q values corresponding to the actions taken
                action_batch = action_batch.max(dim=-1)[1].to(self.device)
                # import pdb; pdb.set_trace()
                q_values = torch.gather(q_batch, 2, action_batch.unsqueeze(dim=-1))
                # q_values is a column vector containing q values for the actions specified by action_batch
                return q_values, new_hidden_batch
        else:
            # if no action specified return all q values
            return q_batch, new_hidden_batch

    def get_actions(self, observation_batch, prev_action_batch, hidden_states, t_env, available_actions=None, explore=True, warmup=False):
        """
        get actions in epsilon-greedy manner, if specified
        """

        if len(observation_batch.shape) == 2:
            batch_size = observation_batch.shape[0]
            no_sequence = True
        else:
            batch_size = observation_batch.shape[1]
            seq_len = observation_batch.shape[0]
            no_sequence = False

        q_values, new_hidden_states = self.get_q_values(observation_batch, prev_action_batch, hidden_states)
        # mask the available actions by giving -inf q values to unavailable actions
        if available_actions is not None:
            q_values[available_actions == 0.0] = -1e10
        #greedy_Qs, greedy_actions = list(map(lambda a: a.max(dim=-1), q_values))
        if self.multidiscrete:
            onehot_actions = []
            greedy_Qs = []
            for i in range(len(self.act_dim)):
                greedy_Q, greedy_action = q_values[i].max(dim=-1)

                if explore:
                    assert no_sequence, "Can only explore on non-sequences"
                    if warmup:
                        eps = 1.0
                    else:
                        eps = self.schedule.eval(t_env)
                    # rand_like samples from Uniform[0, 1]
                    rand_number = torch.rand_like(observation_batch[:, 0])
                    # random actions sample uniformly from action space
                    random_action = Categorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().to(self.device)              
                    take_random = (rand_number < eps).float()
                    
                    action = (1.0 - take_random) * greedy_action.float() + take_random * random_action
                    onehot_action = make_onehot(action.cpu(), batch_size, self.act_dim[i]).to(self.device).detach()
                else:
                    greedy_Q = greedy_Q.unsqueeze(-1)               
                    if no_sequence:
                        onehot_action = make_onehot(greedy_action.cpu(), batch_size, self.act_dim[i])
                    else:
                        onehot_action = make_onehot(greedy_action.cpu(), batch_size, self.act_dim[i],seq_len=seq_len)
                
                onehot_actions.append(onehot_action)
                greedy_Qs.append(greedy_Q)

            onehot_actions = torch.cat(onehot_actions,dim=-1)
            greedy_Qs = torch.cat(greedy_Qs, dim=-1)

            if explore:
                return onehot_actions, greedy_Qs, new_hidden_states.detach(), eps
            else:
                return onehot_actions, greedy_Qs, new_hidden_states, None
        else:
            greedy_Qs, greedy_actions = q_values.max(dim=-1)
            if explore:
                assert no_sequence, "Can only explore on non-sequences"
                if warmup:
                    eps = 1.0
                else:
                    eps = self.schedule.eval(t_env)
                # rand_like samples from Uniform[0, 1]
                rand_numbers = torch.rand_like(observation_batch[:, 0])
                # random actions sample uniformly from action space
                logits = torch.ones_like(prev_action_batch)
                random_actions = avail_choose(logits, available_actions).sample().to(self.device).float()
                take_random = (rand_numbers < eps).float()
                
                actions = (1.0 - take_random) * greedy_actions.float() + take_random * random_actions 
                return make_onehot(actions.cpu(), batch_size, self.act_dim).to(self.device).detach(), greedy_Qs, new_hidden_states.detach(), eps

            else:
                greedy_Qs = greedy_Qs.unsqueeze(-1)
                if no_sequence:
                    return make_onehot(greedy_actions, batch_size, self.act_dim), greedy_Qs, new_hidden_states, None
                else:
                    return make_onehot(greedy_actions, batch_size, self.act_dim,
                                    seq_len=seq_len), greedy_Qs, new_hidden_states, None

    def init_hidden(self, num_agents, batch_size):
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size).to(self.device)
        else:
            return torch.zeros(num_agents, batch_size, self.hidden_size).to(self.device)

    def parameters(self):
        return self.q_network.parameters()

    def load_state(self, source_policy):
        self.q_network.load_state_dict(source_policy.q_network.state_dict())
