#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_players=2
algo="qmix/qmix"
seed_max=1

echo "env is ${env}, hanabi game is ${hanabi}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python run_hanabi_qmix.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_players ${num_players} --seed ${seed} --num_env_steps 100000000 --batch_size 32 --hidden_size 512 --layer_N 2 --lr 0.000025 --opti_eps 0.00003125 --buffer_size 50000 --train_interval_episode 4 --hard_update_interval_episode 500 --epsilon_anneal_time 8000 --epsilon_finish 0.0
    echo "training is done!"
done
