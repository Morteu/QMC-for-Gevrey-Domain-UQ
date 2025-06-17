#!/bin/bash

#SBATCH --job-name=Disk_128
#SBATCH --mail-user=$USER@fu-berlin.de
#SBATCH --mail-type=END
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=1700
#SBATCH --time=80:00:00
#SBATCH --array=1-12
#SBATCH --qos=standard

module load Anaconda3/2022.05
source activate fenicsenv

n_values=(67 127 251 503 1009 2003 4001 8009 16007 32003 64007 128021)

n=${n_values[$SLURM_ARRAY_TASK_ID-1]} 

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 3))

path=/home/$USER/Code_disk/Results/$SLURM_JOB_ID
mkdir $path
cp /home/$USER/Code_disk/inputs/y_sample.npy $path

python /home/$USER/Code_disk/convergence.py --n $n --folder $path 


