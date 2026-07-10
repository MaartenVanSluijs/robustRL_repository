#!/bin/bash

envs=("Hopper-v5" "Walker2d-v5")
target_kls=(0.03 0.003)
clip_ratios=(0.1 0.2 0.3)
lambdas=(0.9 0.95 1.0)

# Loop over parameter combinations and submit Slurm jobs
for env in "${envs[@]}"; do

    for target_kl in "${target_kls[@]}"; do

        for clip_ratio in "${clip_ratios[@]}"; do

            for lambda in "${lambdas[@]}"; do

                echo "Training agent on $env with params (target_kl: $target_kl, clip_ratio:$clip_ratio, lambda:$lambda)"
                sbatch job_script.sh $env $target_kl $clip_ratio $lambda 

            done
        done
    done
done