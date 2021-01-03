#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_players=2
algo="rmasac/rmasac"
seed_max=1

echo "env is ${env}, hanabi game is ${hanabi}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python run_hanabi_rmasac.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_players ${num_players} --seed ${seed} --test_interval 100 --num_env_steps 2000000 --batch_size 32 --hidden_size 512 --layer_N 2 --gain 0.01 --lr 7e-4 --buffer_size 5000 --hard_update_interval_episode 200 --tau 0.01
    echo "training is done!"
done
