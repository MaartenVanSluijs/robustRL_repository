#!/bin/bash

seeds=(10 34 40 41 53 68 71 72 96 98)
env=Hopper-v5
target_kl=0.03
clip_ratio=0.3
lambda=0.9

# Loop over parameter combinations and submit Slurm jobs
for seed in "${seeds[@]}"; do
    echo "Training agent on $env with seed $seed)"
    job_script.sh $env $target_kl $clip_ratio $lambda $seed 
done