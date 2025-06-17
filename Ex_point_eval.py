from domain_generation.reference_domain import Ellipse, Hexagon, Square
from FEM import create_mesh, solve_poisson
from inverse import point_evaluation
from QMC_par import Experiment
from numpy import abs, load
from domain_generation.sample import sample
import argparse

def run_simulation(n):
    '''Runs an experiment for a single value of n. Useful for job arrays when using the HPC system'''

    s = 20
    domain = Ellipse(num_points=128, a=1, b=1)

    Dref = "sampled_domain"

    if Dref == "sampled_domain":
        y = load('Miscellanea/y_sample.npy')
        og_domain = Ellipse(num_points=256, a=1, b=1)
        DwX, DwY = sample(og_domain, len(y), y)
        mesh = create_mesh(DwX, DwY, max_angle=7)
    elif Dref == "square":
        square = Square(num_points=128, a=1.8)
        square_x, square_y = square.points
        mesh = create_mesh(square_x, square_y, max_angle=7)

    uh, V = solve_poisson(mesh)
    delta = point_evaluation(uh)
    gamma = 0.1*(abs(delta).max()) # noise is 10% of max observation

    nlist = [n]
    q = 8
 
    experiment = Experiment(domain, s, delta, gamma)
    experiment.get_results(nlist, q)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with different n values.")
    parser.add_argument("--n", type=int, required=True, help="Value of n to run the simulation.")
    args = parser.parse_args()
    
    run_simulation(args.n)