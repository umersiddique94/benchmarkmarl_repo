#!/bin/sh
env="StarCraft2"
map="3s_vs_4z"
algo="rmasac_tau0.005"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=4 python run_sc_rmasac.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed 5 --tau 0.005
    echo "training is done!"
done
