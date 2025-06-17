#!/bin/bash

#SBATCH --job-name=Disk_last_128
#SBATCH --mail-user=$USER@fu-berlin.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=1800
#SBATCH --time=120:00:00
#SBATCH --qos=standard

module load Anaconda3/2022.05
source activate fenicsenv

n=128021

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 3))

python /home/$USER/Code_disk/convergence.py --n $n


