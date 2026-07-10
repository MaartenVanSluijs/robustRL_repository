#!/bin/bash

#SBATCH --job-name=test_run
#SBATCH --output=job_outputs/output_%j.txt
#SBATCH --partition=tue.default.q
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

# Load modules or software if needed
module load Python/3.11.3-GCCcore-12.3.0

#set MuJoCo rendering thing
export MUJOCO_GL=egl
export WANDB_API_KEY=8aebc986acc63ba64fc024ead807f0b0c6d333b5

source ./env/bin/activate
wandb login

# Execute the script or command
python ddpg.py --env_name $1 --noise_scale $2 --batch_size $3 --polyak $4 --seed $5 --log_results True --num_epochs 1500