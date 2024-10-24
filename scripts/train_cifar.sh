#!/bin/bash
#SBATCH --job-name="train_cifar10"   # Sensible name for the job
#SBATCH --account=ie-idi      # Account for consumed resources
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)

module load Python/3.10.15-GCCcore-14.2.0

python examples/images/cifar10/train_cifar10.py