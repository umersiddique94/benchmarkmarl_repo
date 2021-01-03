#!/bin/sh
env="StarCraft2"
map="2s_vs_1sc"
algo="rmaddpg_tau0.005"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python run_sc_rmaddpg.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --tau 0.005
    echo "training is done!"
done
