#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_players=2
algo="rmatd3/rmatd3"
seed_max=1

echo "env is ${env}, hanabi game is ${hanabi}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python run_hanabi_rmatd3.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_players ${num_players} --seed ${seed} --num_env_steps 100000000 --batch_size 32 --hidden_size 64 --layer_N 1 --lr 0.000025 --opti_eps 0.00003125 --buffer_size 500 --train_interval_episode 1 --actor_train_interval_episode 2 --epsilon_anneal_time 80000 --epsilon_finish 0.0
    echo "training is done!"
done
