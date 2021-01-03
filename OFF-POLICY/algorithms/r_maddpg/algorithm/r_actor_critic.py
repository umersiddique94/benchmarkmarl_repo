import torch
import torch.nn as nn
import copy
import numpy as np
from algorithms.common.common_utils import init

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        if use_orthogonal:
            if use_ReLU:
                active_func = nn.ReLU()
                init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()
                init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))
        else:
            if use_ReLU:
                active_func = nn.ReLU()
                init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()
                init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), gain = nn.init.calculate_gain('tanh'))

        self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)
    
    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class R_Actor(nn.Module):
    def  __init__(self, args, obs_dim, act_dim, discrete_action, device, take_prev_action=False):
        super(R_Actor, self).__init__()
        self._use_feature_normlization = args.use_feature_normlization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = args.hidden_size
        self.discrete = discrete_action
        self.device = device
        self.take_prev_act = take_prev_action

        if take_prev_action:
            input_dim = obs_dim + act_dim
        else:
            input_dim = obs_dim

        # map observation input into input for rnn
        if self._use_feature_normlization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)
        self.mlp = MLPLayer(input_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU).to(self.device)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size).to(self.device)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(self.hidden_size).to(self.device)
        # get action from rnn hidden state
        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), self._gain)
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0), self._gain)

        if isinstance(act_dim, np.ndarray):
            # MultiDiscrete setting: have n Linear layers for each action
            self.multidiscrete = True
            self.action_outs = [init_(nn.Linear(self.hidden_size, a_dim)).to(self.device) for a_dim in act_dim]
        else:
            self.multidiscrete = False
            self.action_out = init_(nn.Linear(self.hidden_size, act_dim)).to(self.device)

    def init_hidden(self, batch_size):
        hidden = self.mlp.fc1[0].weight.new_zeros(batch_size, self.hidden_size)
        return hidden

    def forward(self, obs, prev_acts, rnn_hidden_states):
        # make sure input is a torch tensor
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).float()
        if prev_acts is not None and type(prev_acts) == np.ndarray:
            prev_acts = torch.from_numpy(prev_acts).float()
        if type(rnn_hidden_states) == np.ndarray:
            rnn_hidden_states = torch.from_numpy(rnn_hidden_states).float()
        
        #assert prev_acts == None or len(obs.shape) == len(prev_acts.shape)

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            if self.take_prev_act:
                prev_acts = prev_acts[None]
            # obs is now of shape (seq_len, batch_size, obs_dim)

        if self.take_prev_act:
            inp = torch.cat((obs, prev_acts), dim=-1)
        else:
            inp = obs

        if len(rnn_hidden_states.shape) == 2:
            # hiddens should be of shape (1, batch_size, dim)
            rnn_hidden_states = rnn_hidden_states[None]
        
        inp = inp.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)

        # get RNN input
        if self._use_feature_normlization:
            inp = self.feature_norm(inp)
        rnn_inp = self.mlp(inp)
        # pass RNN input and hidden states through RNN to get the hidden state sequence and the final hidden
        self.rnn.flatten_parameters()
        rnn_outs, h_final = self.rnn(rnn_inp, rnn_hidden_states)
        rnn_outs = self.norm(rnn_outs)
        # pass outputs through linear layer # TODO: should i put a activation in between this??

        if self.multidiscrete:
            act_outs = []
            for a_out in self.action_outs:
                act_out = a_out(rnn_outs)
                if no_sequence:
                # remove the dummy first time dimension if the input didn't have a time dimension
                    act_out = act_out[0, :, :]
                act_outs.append(act_out.cpu())
        else:
            act_outs = self.action_out(rnn_outs)
            if no_sequence:
                # remove the dummy first time dimension if the input didn't have a time dimension
                act_outs = act_outs[0, :, :]
            act_outs = act_outs.cpu()

        h_final = h_final.cpu()

        return act_outs, h_final[0, :, :]

class R_Critic(nn.Module):
    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(R_Critic, self).__init__()
        self._use_feature_normlization = args.use_feature_normlization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self.central_obs_dim = central_obs_dim
        self.central_act_dim = central_act_dim
        self.hidden_size = args.hidden_size
        self.device = device

        input_dim = central_obs_dim + central_act_dim
        if self._use_feature_normlization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)

        # map observation input into input for rnn
        self.mlp = MLPLayer(input_dim, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU).to(self.device)  
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size).to(self.device)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

        self.norm = nn.LayerNorm(self.hidden_size).to(self.device)
        if self._use_orthogonal:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        else:
            init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(x, 0))
        
        self.q_out = init_(nn.Linear(self.hidden_size, 1)).to(self.device)

    def init_hidden(self, batch_size):
        return self.mlp.fc1[0].weight.new_zeros(batch_size, self.hidden_size)

    def forward(self, central_obs, central_act, rnn_hidden_states):
        # ensure inputs are torch tensors
        if type(central_obs) == np.ndarray:
            central_obs = torch.from_numpy(central_obs).float()
        if type(central_act) == np.ndarray:
            central_act = torch.from_numpy(central_act).float()
        if type(rnn_hidden_states) == np.ndarray:
            rnn_hidden_states = torch.from_numpy(rnn_hidden_states).float()

        no_sequence = False
        if len(central_obs.shape) == 2 and len(central_act.shape) == 2:
            # no sequence, so add a time dimension of len 0
            no_sequence = True
            central_obs, central_act = central_obs[None], central_act[None]

        if len(rnn_hidden_states.shape) == 2:
            # also add a first dimension to the rnn hidden states
            rnn_hidden_states = rnn_hidden_states[None]

        central_obs = central_obs.float().to(self.device)
        central_act = central_act.float().to(self.device)
        rnn_hidden_states = rnn_hidden_states.float().to(self.device)
        
        x = torch.cat([central_obs, central_act], dim=2)
        if self._use_feature_normlization:
            x = self.feature_norm(x)
        rnn_inp = self.mlp(x)
        self.rnn.flatten_parameters()
      
        rnn_outs, h_final = self.rnn(rnn_inp, rnn_hidden_states)
        rnn_outs = self.norm(rnn_outs)
        q_values = self.q_out(rnn_outs)

        if no_sequence:
            # remove the time dimension
            q_values = q_values[0, :, :]

        h_final = h_final[0, :, :]

        q_values = q_values.cpu()
        h_final = h_final.cpu()

        return q_values, h_final