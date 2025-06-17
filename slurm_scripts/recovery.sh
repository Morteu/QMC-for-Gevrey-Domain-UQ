#!/bin/bash

#SBATCH --job-name=shape_recovery_sampled
#SBATCH --mail-user=$USER@fu-berlin.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=1000
#SBATCH --time=10:00:00
#SBATCH --array=1-12
#SBATCH --qos=standard

module load Anaconda3/2022.05
source activate fenicsenv

n_values=(67 127 251 503 1009 2003 4001 8009 16007 32003 64007 128021)

n=${n_values[$SLURM_ARRAY_TASK_ID-1]} 

export OMP_NUM_THREADS=7

python /home/$USER/Code_disk/ex_point_eval.py --n $n


