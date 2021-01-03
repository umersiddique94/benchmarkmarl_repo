#!/bin/sh
env="StarCraft2"
map="2c_vs_64zg"
algo="rmatd3_tau0.005"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python run_sc_rmatd3.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --tau 0.005
    echo "training is done!"
done
