import sys
import os
import time
import psutil
from memory_profiler import memory_usage

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from FEM import create_mesh, solve_poisson
from domain_generation.sample import sample
from domain_generation.perturbation import get_V_exp
from domain_generation.reference_domain import Ellipse, Hexagon, Square
from inverse import observe_solution_simpsons, evaluate_points
from datetime import datetime
from numpy import random, load, abs, save
from QMC_par_shifts import Experiment

''' 
Used for timing different important processes to spot bottlenecks and bugs, 
as well as estimating memory and CPU requirements.
'''

process = psutil.Process(os.getpid())

def solve_and_time():
    s = 100
    start_time = datetime.now()
    domain = Ellipse(num_points=128, a=1, b=1)
    # domain = Hexagon(num_points=100, l=1)
    y = random.uniform(-0.5, 0.5, size=s)
    DwX, DwY = sample(domain, s, y)
    sample_end = datetime.now()
    mesh = create_mesh(DwX, DwY, min_angle=7)
    mesh_end = datetime.now()
  
    uh, V = solve_poisson(mesh)
    solve_end = datetime.now()

    # observation = observe_solution_simpsons(uh, R=0.3, n_r=25, n_theta=301)
    og_eval_xs, og_eval_ys = domain.evaluation_points
    eval_xs, eval_ys = get_V_exp(og_eval_xs, og_eval_ys, y, s) 
    observation = evaluate_points(uh, eval_xs, eval_ys)
    end_time = datetime.now()
    
    print(f'Value of the observation: {observation}\n')

    sample_time = (sample_end - start_time).total_seconds()
    mesh_time = (mesh_end - sample_end).total_seconds()
    solve_time = (solve_end - mesh_end).total_seconds()
    observation_time = (end_time - solve_end).total_seconds()
    total_time = (end_time - start_time).total_seconds()

    sample_prc = round((sample_time/total_time)*100, 1)
    mesh_prc = round((mesh_time/total_time)*100, 1)
    solve_prc = round((solve_time/total_time)*100, 1)
    observe_prc = round((observation_time/total_time)*100, 1)

    print(f"Sampling: {sample_time}s --> {sample_prc}%")
    print(f"Creating mesh: {mesh_time}s --> {mesh_prc}%")
    print(f"Solving PDE: {solve_time}s --> {solve_prc}%")
    print(f"Observing solution: {observation_time}s --> {observe_prc}%")
    print(f"Total time: {total_time}s\n")


# Function to simulate your workload
def simulate_workload(n):
    start_time = time.time()
    
    s = 100
    domain = Ellipse(num_points=128, a=1, b=1)
    og_eval_xs, og_eval_ys = domain.evaluation_points
    domain_x, domain_y = domain.points
    mesh = create_mesh(domain_x, domain_y, min_angle=7)

    # For reconstruction:
    # Dref = "sampled_domain"
    # if Dref == "sampled_domain":
    #     y = load('Miscellanea/y_sample.npy')
    #     og_domain = Ellipse(num_points=256, a=1, b=1)
    #     DwX, DwY = sample(og_domain, len(y), y)
    #     mesh = create_mesh(DwX, DwY, max_angle=7)
    # elif Dref == "square":
    #     square = Square(num_points=128, a=1.8)
    #     square_x, square_y = square.points
    #     mesh = create_mesh(square_x, square_y, max_angle=20)

    uh, V = solve_poisson(mesh)
    delta = evaluate_points(uh, og_eval_xs, og_eval_ys)
    gamma = 0.10*(abs(delta).max()) # noise level

    nlist = [n]
    q = 8
    # q = 1 # this is enough when we're only interested in the reconstruction

    experiment_1 = Experiment(domain, s, delta, gamma)
    experiment_1.get_results('Results', nlist, q)
    
    end_time = time.time()
    
    # Log the time taken
    execution_time = end_time - start_time
    print(f"Simulation for n={n} took {execution_time:.2f} seconds")

    return execution_time

# Monitor memory usage before and after the workload
def profile_simulation(n):
    print(f"Running simulation for n={n}")

    # Measure memory usage
    mem_usage_before = process.memory_info().rss / 1024 ** 2  # in MB
    print(f"Memory usage before: {mem_usage_before:.2f} MB")

    # Run the simulation and measure execution time
    execution_time = simulate_workload(n)
    
    # Measure CPU usage (in terms of % of a single core)
    cpu_usage = process.cpu_percent(interval=None)  # over the entire process lifetime
    
    # Measure memory usage after
    mem_usage_after = process.memory_info().rss / 1024 ** 2  # in MB
    print(f"Memory usage after: {mem_usage_after:.2f} MB")
    print(f"Peak memory usage during simulation: {max(mem_usage_before, mem_usage_after):.2f} MB")
    print(f"CPU usage during simulation: {cpu_usage:.2f}% of a single core\n")

    return execution_time, cpu_usage, max(mem_usage_before, mem_usage_after)

if __name__ == '__main__':
    # solve_and_time()
    
    # Loop over different values of n
    # ns = [11, 23, 31, 43, 53, 61, 71, 83, 101, 211, 307, 401, 503, 601, 701, 809, 907, 1009, 1091, 1187, 1283, 1381, 1481, 1567, 1657, 1753, 1871, 1979]
    # ns = [61, 127, 251, 503, 1009]
    # ns = [20 03]
    ns = [13, 17, 23]
    
    y = random.uniform(-0.5, 0.5, size=256)
    # save('inputs/y_sample.npy', y)

    for n in ns:
        profile_simulation(n)

