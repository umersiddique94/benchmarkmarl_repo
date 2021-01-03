#!/bin/sh
env="MPE"
scenario="simple_spread"
algo="qmix/tune_hard900_lr5e-4"
seed_max=10

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python run_mpe_qmix.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --seed ${seed} --num_env_steps 2000000 --episode_length 25 --hard_update_interval_episode 900 --lr 5e-4 --epsilon_finish 0.0
    echo "training is done!"
done
