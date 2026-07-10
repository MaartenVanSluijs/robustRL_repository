#!/bin/bash

seeds=(1 11 15 30 52 62 64 86 92 95)
env=Hopper-v5
noise_scale=0.15
batch_size=100
polyak=0.999

# Loop over parameter combinations and submit Slurm jobs
for seed in "${seeds[@]}"; do
    echo "Training agent on $env with seed $seed)"
    sbatch job_script.sh $env $noise_scale $batch_size $polyak $seed 
done