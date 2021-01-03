#!/bin/sh
env="HNS"
hns_name="BoxLocking"
algo="rmaddpg/rmaddpg"
seed_max=1

echo "env is ${env}, hns name is ${hns_name}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python run_hns_rmaddpg.py --env_name ${env} --algorithm_name ${algo} --hns_name ${hns_name} --seed ${seed} --episode_length 120 --batch_size 256 --chunk_len 120 --buffer_size 5000 --num_env_steps 50000000 --gain 0.01 --tau 0.005 --lr 5e-4
    echo "training is done!"
done
