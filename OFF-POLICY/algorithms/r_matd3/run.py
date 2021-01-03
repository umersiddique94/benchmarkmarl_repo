import argparse


def parse_args(args):
    alg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    alg_parser.add_argument('--env_name', type=str, default="Starcraft2")
    alg_parser.add_argument("--algorithm_name", type=str, default='rmatd3')
    alg_parser.add_argument("--model_dir", type=str, default=None)

    # Episode parameters
    alg_parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for env")
    alg_parser.add_argument('--episode_length', type=int, default=80, help="Max length for any episode")
    alg_parser.add_argument('--buffer_size', type=int, default=5000, help="Max # of transitions that replay buffer can contain")

    # optimization/training parameters
    alg_parser.add_argument('--share_policy', action='store_false', default=True, help="Whether use a centralized critic")
    alg_parser.add_argument('--decentralized', action='store_true', default=False, help="Whether use a centralized critic")
    alg_parser.add_argument('--use_feature_normlization', action='store_true', default=False, help="Whether to apply layernorm to the inputs")
    alg_parser.add_argument('--use_orthogonal', action='store_false', default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    alg_parser.add_argument('--use_ReLU', action='store_false', default=True, help="Whether to use ReLU")
    alg_parser.add_argument('--layer_N', type=int, default=1, help="Number of layers for actor/critic networks")
    alg_parser.add_argument('--hidden_size', type=int, default=64, help="Dimension of hidden layers for actor/critic networks")
    alg_parser.add_argument("--opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    alg_parser.add_argument("--weight_decay", type=float, default=0)
    alg_parser.add_argument("--gain", type=float, default=1)
    
    
    alg_parser.add_argument('--chunk_len', type=int, default=80, help="Time length of chunks used to train via BPTT")
    alg_parser.add_argument('--burn_in_time', type=int, default=0, help="Length of burn in time for RNN training, see R2D2 paper")
    alg_parser.add_argument('--num_env_steps', type=int, default=2000000, help="Number of env steps to train for")
    alg_parser.add_argument('--batch_size', type=int, default=32, help="Number of buffer transitions to train on at once")
    alg_parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate for Adam")
    alg_parser.add_argument('--grad_norm_clip', type=float, default=10.0, help="Max gradient norm (clipped if above this value)")
    alg_parser.add_argument('--tau', type=float, default=0.01, help="Polyak update rate")
    alg_parser.add_argument('--huber_loss', type=bool, default=False, help="Whether to use Huber loss for critic update")
    alg_parser.add_argument('--prev_act_inp', action='store_true', default=False, help="Whether the actor input takes in previous actions as part of its input")
    alg_parser.add_argument('--automatic_entropy_tune', action='store_false', default=True, help="Whether use a centralized critic")
    
    # exploration parameters
    alg_parser.add_argument('--num_random_episodes', type=int, default=5, help="Number of episodes to add to buffer with purely random actions")
    alg_parser.add_argument('--epsilon_start', type=float, default=1.0, help="Starting value for epsilon, for eps-greedy exploration")
    alg_parser.add_argument('--epsilon_finish', type=float, default=0.0, help="Ending value for epsilon, for eps-greedy exploration")
    alg_parser.add_argument('--epsilon_anneal_time', type=int, default=5000, help="Number of episodes until epsilon reaches epsilon_finish")
    alg_parser.add_argument('--act_noise_std', type=float, default=0.1, help="Action noise")

    # logging parameters
    alg_parser.add_argument('--actor_train_interval_episode', type=int, default=2, help="After how many critic updates actor should be updated")
    alg_parser.add_argument('--train_interval_episode', type=int, default=1, help="Number of env steps between updates to actor/critic")
    alg_parser.add_argument('--train_interval', type=int, default=100, help="Number of episodes between updates to actor/critic")
    alg_parser.add_argument('--test_interval', type=int,  default=10000, help="After how many episodes the policy should be tested")
    alg_parser.add_argument('--save_interval', type=int, default=50000, help="After how many episodes of training the policy model should be saved")
    alg_parser.add_argument('--log_interval', type=int, default=1000, help="After how many episodes of training the policy model should be saved")
    alg_parser.add_argument('--num_test_episodes', type=int, default=32, help="How many episodes to collect for each test")

    # run parameters
    alg_parser.add_argument('--n_training_threads', type=int,  default=10, help="Number of torch threads for training")
    alg_parser.add_argument('--n_rollout_threads', type=int,  default=32, help="Number of torch threads for training")
    alg_parser.add_argument('--seed', type=int, default=0, help="Random seed for numpy/torch")
    alg_parser.add_argument("--cuda", action='store_false', default=True)
    alg_parser.add_argument("--cuda_deterministic", action='store_false', default=True)

    parsed_args = alg_parser.parse_known_args(args)[0]

    return parsed_args, vars(parsed_args)