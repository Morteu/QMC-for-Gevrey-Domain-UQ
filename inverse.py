from scipy.stats import norm
import FEM
from domain_generation.sample import sample
from numpy import linspace, ones, cos, sin, pi, array, ndarray, random, linalg
from domain_generation.perturbation import get_V_exp

def observe_solution_simpsons(uh, R=0.3, n_r=11, n_theta=301):
    """
    Numerically approximates the integral of the discretized solution uh(x, y) over a disk of radius R using Simpson's rule.
    
    Args:
        uh (fenics solution): function that takes a tuple (x, y) and returns the value at that point
        R (float): radius of the disk
        n_r (int): (odd) number of points in the radial direction (must be odd for Simpson's rule)
        n_theta (int): (odd) number of points in the angular direction (must be odd for Simpson's rule)
    
    Returns:
        inter_value (float): The integral of uh normalized by the area of the disk.
    """
    # Radial and angular grids
    r = linspace(0, R, n_r)
    theta = linspace(0, 2 * pi, n_theta)
    dr = r[1] - r[0] 
    dtheta = theta[1] - theta[0]

    # Precompute weights for Simpson's rule
    radial_weights = ones(n_r)
    radial_weights[1:-1:2] = 4
    radial_weights[2:-2:2] = 2 
    radial_weights[0] = radial_weights[-1] = 1 

    # Simpson's rule weights in angular direction
    angular_weights = ones(n_theta)
    angular_weights[1:-1:2] = 4  
    angular_weights[2:-2:2] = 2  
    angular_weights[0] = angular_weights[-1] = 1

    integral = 0.0
    for i, ri in enumerate(r):
        for j, tj in enumerate(theta):
            x = ri * cos(tj)
            y = ri * sin(tj)
            integral += uh((x, y)) * ri * radial_weights[i] * angular_weights[j]

    # Simpson's rule normalizes by (dx/3) and (dy/3) in 1D, so here we apply both
    integral *= (dr / 3) * (dtheta / 3)

    # Normalize by the area of the disk (πR²)
    inter_value = integral / (pi * R**2)
    
    return inter_value

def evaluate_points(uh, xs, ys):
    observations = array([uh(x, y) for x, y in zip(xs, ys)])
        
    return observations

def sample_delta(min_value, max_value):
    delta = random.uniform(low=min_value, high=max_value)
    return delta

def get_EV_integrands(domain, s, y, delta, gamma):
    DwX, DwY = sample(domain, s, y)
    mesh = FEM.create_mesh(DwX, DwY, min_angle=7)
    uh, V = FEM.solve_poisson(mesh)
    og_eval_xs, og_eval_ys = domain.evaluation_points
    eval_xs, eval_ys = get_V_exp(og_eval_xs, og_eval_ys, y, s) 

    if isinstance(delta, ndarray):
        G = evaluate_points(uh, eval_xs, eval_ys)
        nu_delta = norm.pdf(linalg.norm(delta - G), loc=0, scale=gamma) 
    else:
        G = observe_solution_simpsons(uh, R=0.3, n_r=15, n_theta=301)  
        nu_delta = norm.pdf(delta - G, loc=0, scale=gamma)
    
    return nu_delta

if __name__ == "__main__":
    evaluate_points(1, 50, 0.25)