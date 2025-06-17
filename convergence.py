from domain_generation.reference_domain import Ellipse, Square
from FEM import create_mesh, solve_poisson
from QMC_par import Experiment
from domain_generation.sample import sample
from domain_generation.perturbation import *
from numpy import abs, load
from inverse import evaluate_points
import argparse
from utils.my_logger import log_parameters
import os

'''
Runs an experiment to obtain convergence results for the RMS error
'''

def run_simulation(n):
    '''Runs an experiment for a single value of n. Useful for job arrays when using the HPC system'''

    s = 100
    domain = Ellipse(num_points=128, a=1, b=1)

    # Define "original" domain (s = 200, h = 2^-6) used to generate data and solve Poisson eq
    y = load('inputs/y_sample.npy')
    og_domain = Ellipse(num_points=256, a=1, b=1)
    DwX, DwY = sample(og_domain, len(y), y)
    mesh = create_mesh(DwX, DwY, max_angle=7)
    uh, V = solve_poisson(mesh)

    # Find evaluation points for original domain
    ref_eval_xs, ref_eval_ys = og_domain.evaluation_points
    og_eval_xs, og_eval_ys = get_V_exp(ref_eval_xs, ref_eval_ys, y, len(y)) 
    delta = evaluate_points(uh, og_eval_xs, og_eval_ys)

    noise_level = 0.1
    gamma = noise_level*(abs(delta).max())
    nlist = [n]
    q = 8
 
    experiment = Experiment(domain, s, delta, gamma)
    experiment.get_results(args.folder, nlist, q)

    file_path = os.path.join(args.folder, 'parameters.txt')
    if args.folder is not None and not os.path.exists(file_path):
        log_parameters(args.folder, s, domain, og_domain, len(y), noise_level, q)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with different n values.")
    parser.add_argument("--n", type=int, required=True, help="Value of n to run the simulation.")
    parser.add_argument("--folder", type=str, required=False, help="Path to store the results (JOB_ID if run from cluster)")
    args = parser.parse_args()
    
    run_simulation(args.n)