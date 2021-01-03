#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=2
algo="small2/mappo-gru"
seed_max=3
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_rollout_threads 1000 --n_training_threads 1 --num_mini_batch 5 --episode_length 80 --num_env_steps 100000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --hidden_size 512 --layer_N 2 
    echo "training is done!"
done
