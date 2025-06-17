#!/bin/bash

#SBATCH --job-name=Disk_128
#SBATCH --mail-user=$USER@fu-berlin.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=500
#SBATCH --time=00:10:00
#SBATCH --array=1-2
#SBATCH --qos=standard

module load Anaconda3/2022.05
source activate fenicsenv

n_values=(11 23)

n=${n_values[$SLURM_ARRAY_TASK_ID-1]} 

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 3))

python /home/$USER/Code_disk/convergence.py --n $n