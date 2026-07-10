#!/bin/bash

envs=("Hopper-v5" "Walker2d-v5")
noise_scales=(0.05 0.1 0.15)
batch_sizes=(100 150 200)
polyaks=(0.99 0.999)

# Loop over parameter combinations and submit Slurm jobs
for env in "${envs[@]}"; do

    for noise_scale in "${noise_scales[@]}"; do

        for batch_size in "${batch_sizes[@]}"; do

            for polyak in "${polyaks[@]}"; do

                echo "Training agent on $env with params (noise_scale:$noise_scale, batch_size:$batch_size, polyak:$polyak)"
                sbatch job_script.sh $env $noise_scale $batch_size $polyak 

            done
        done
    done
done