import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
#from envs.multiagent_particle_envs.mpe.multiagent.multi_discrete import MultiDiscrete as MultiDiscreteMPE
import copy

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
    
# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, available_actions = None, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    if type(logits) == np.ndarray:
        logits = torch.from_numpy(logits)
    if type(available_actions) == np.ndarray:
        available_actions = torch.from_numpy(available_actions)

    dim = len(logits.shape) - 1
    logits_avail = logits.clone()
    logits_avail[available_actions==0]=-1e10   
    argmax_acs = (logits_avail == logits_avail.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, device = 'cpu'):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    if device == 'cpu':
        y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    else:
        y = (logits.cpu() + sample_gumbel(logits.shape, tens_type=type(logits.data))).cuda()

    dim = len(logits.shape) - 1
    return F.softmax(y / temperature, dim=dim)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, available_actions=None, temperature=1.0, hard=False, device = 'cpu'):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, device)
    if hard:       
        y_hard = onehot_from_logits(y, available_actions)
        y = (y_hard - y).detach() + y
    return y

def gaussian_noise(shape, std):
    return torch.empty(shape).normal_(mean=0, std=std)

def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, Discrete):
        dim = space.n
    elif isinstance(space, Tuple):
        dim = sum([get_dim_from_space(sp) for sp in space])
    elif isinstance(space, MultiDiscrete):
        # TODO: support multidiscrete spaces
        return (space.high - space.low) + 1

    elif isinstance(space, list):
        dim = space[0]
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim

def get_state_dim(observation_dict, action_dict):
    combined_obs_dim = sum([get_dim_from_space(space) for space in observation_dict.values()])
    combined_act_dim = 0
    for space in action_dict.values():
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            combined_act_dim += int(sum(dim))
        else:
            combined_act_dim += dim
    return combined_obs_dim, combined_act_dim, combined_obs_dim+combined_act_dim

def is_discrete(space):
    if isinstance(space, Discrete) or isinstance(space, MultiDiscrete):
        return True
    else:
        return False


class DecayThenFlatSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

def make_onehot(ind_vector, batch_size, action_dim, seq_len=None):
    if not seq_len:
        onehot_mat = torch.zeros((batch_size, action_dim)).float()
        onehot_mat[torch.arange(batch_size), ind_vector.long()] = 1
        return onehot_mat
    if seq_len:
        onehot_mats = []
        for i in range(seq_len):
            mat = torch.zeros((batch_size, action_dim)).float()
            mat[torch.arange(batch_size), ind_vector[i].long()] = 1
            onehot_mats.append(mat)
        return torch.stack(onehot_mats)

def biased_xavier_uniform_weight_init(bias):
    def xavier_weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, bias)
        elif isinstance(m, nn.GRUCell):
            torch.nn.init.xavier_uniform_(m.weight_hh, gain=1)
            torch.nn.init.xavier_uniform_(m.weight_ih, gain=1)

            hidden_dim = m.bias_ih.shape[0] // 3
            torch.nn.init.constant_(m.bias_ih[:hidden_dim], -1)
            torch.nn.init.constant_(m.bias_hh[:hidden_dim], -1)
            torch.nn.init.constant_(m.bias_ih[hidden_dim: ], 0)
            torch.nn.init.constant_(m.bias_hh[hidden_dim: ], 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)


    return xavier_weight_init

def biased_orthogonal_weight_init(bias):
    def orthogonal_weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, bias)
        if isinstance(m, nn.GRUCell):
            torch.nn.init.orthogonal_(m.weight_hh, gain=1)
            torch.nn.init.orthogonal_(m.weight_ih, gain=1)

            hidden_dim = m.bias_ih.shape[0] // 3
            torch.nn.init.constant_(m.bias_ih[:hidden_dim], -1)
            torch.nn.init.constant_(m.bias_hh[:hidden_dim], -1)
            torch.nn.init.constant_(m.bias_ih[hidden_dim: ], 0)
            torch.nn.init.constant_(m.bias_hh[hidden_dim: ], 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
                    
    return orthogonal_weight_init

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample()

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

def avail_choose(x, available_actions=None):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    if type(available_actions) == np.ndarray:
        available_actions = torch.from_numpy(available_actions)

    x[available_actions==0]=-1e10
    return FixedCategorical(logits=x)

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]
    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)
