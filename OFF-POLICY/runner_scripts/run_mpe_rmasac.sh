#!/bin/sh
env="MPE"
scenario="simple_reference"
algo="rmasac/rmasac_tau0.005_lr5e-4_coef0.5_norm"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python run_mpe_rmasac.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --seed 78 --num_env_steps 2000000 --episode_length 25 --tau 0.005 --lr 5e-4 --target_entropy_coef 0.5 --use_feature_normlization
    echo "training is done!"
done
