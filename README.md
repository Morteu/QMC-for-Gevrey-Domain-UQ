# QMC for Bayesian shape inversion subject to Gevrey deformations

This repository hosts the code used to produce the numerical experiments in the paper _Quasi-Monte Carlo for Bayesian shape inversion governed by the Poisson problem subject to Gevrey regular domain deformations_.

## Dependencies

- shapely
- SciPy
- NumPy
- pandas
- matplotlib
- FEniCS (2019)

## Structure of the code

### domain_generation
Contains submodules defining the reference domain and different possible perturbation fields.

### utils
Various useful submodules for logging information, plotting, timing the code, and processing the results.

### FEM.py / FEMx.py
Meshes the domains and calls FEniCS to obtain a discretized solution to Poisson's equation on the sampled domains.

### QMC_par_shifts.py
Implementation of the fast CBC algorithm. Includes the definition of the _Experiment_ class.

### inverse.py
Implements the observation operator and Bayes' formula.

### slurm_scripts
A series of example scripts to run jobs in an HPC system.

### Results
Default folder to store the output files.

### inputs
For the storage of generating vectors.

