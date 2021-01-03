#!/bin/sh
env="MPE"
scenario="simple_reference"
algo="rmaddpg/rmaddpg_tau0.005_lr5e-4"
seed_max=10

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python run_mpe_rmaddpg.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --seed ${seed} --episode_length 25 --lr 5e-4 --tau 0.005 --num_env_steps 5000000 
    echo "training is done!"
done
