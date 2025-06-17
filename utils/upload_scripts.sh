#!/bin/bash

destination_folder="maxorteu@curta.zedat.fu-berlin.de:~/Code_disk"
local_folder="$HOME/desktop/TFM/Code"

# Transfer files to the remote server
scp "$local_folder"/QMC_par.py "$destination_folder"
scp "$local_folder"/FEM.py "$destination_folder"
scp "$local_folder"/inverse.py "$destination_folder"
scp "$local_folder"/convergence.py "$destination_folder"
scp -r "$local_folder"/domain_generation "$destination_folder"
scp -r "$local_folder"/inputs "$destination_folder"
scp -r "$local_folder"/utils "$destination_folder"
scp "$local_folder"/slurm_scripts/convergence.sh "$destination_folder"
scp "$local_folder"/slurm_scripts/extra.sh "$destination_folder"
scp "$local_folder"/slurm_scripts/try.sh "$destination_folder"

echo "\n"
echo "All files transferred successfully to $destination_folder!"