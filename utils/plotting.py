import sys
import os
from matplotlib.patches import Circle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from domain_generation.perturbation import *
from shapely.geometry import Point
import pyvista
from mpl_toolkits.mplot3d import Axes3D
from domain_generation.perturbation import *
from dolfin import fem, mesh, plot
from domain_generation.reference_domain import Ellipse, Hexagon, Torus
import FEM
from domain_generation.sample import sample
import numpy as np
from inverse import evaluate_points, sample_delta
from QMC import Experiment
from FEM import create_mesh

# plt.style.use('ggplot')

"""
Visualization utilities for simulation workflows.

This module provides functions for creating plots and figures used in the simulation
pipeline. These visualizations serve both as tools for debugging intermediate steps 
and as figures suitable for publication.

Dependencies include Matplotlib, PyVista, and Shapely, among others. Functions 
within this module often interact with domain generation, sampling procedures, 
and FEM-based simulations.

Typical uses:
- Visualizing sampled points or perturbed domains
- Inspecting mesh structures or boundary conditions
- Generating reproducible visual outputs for papers

Note: This module assumes that the broader project structure is intact and may 
depend on other custom modules like `domain_generation`, `FEM`, and `inverse`.
"""

def qmc_vs_mc(dim=2, n_samples = 128):
    rng = np.random.default_rng()

    sample = {}

    # QMC Halton
    engine = qmc.Halton(d=dim, seed=rng)
    sample["QMC - Halton"] = [engine.random(n_samples)[:, 0]-0.5, engine.random(n_samples)[:, 1]-0.5]

    MC_x = np.random.uniform(-0.5, 0.5, n_samples)
    MC_y = np.random.uniform(-0.5, 0.5, n_samples)
    sample["MC"] = [MC_x, MC_y]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for i, sample_name in enumerate(sample):
        axs[i].scatter(sample[sample_name][0], sample[sample_name][1], c='red', s=20, alpha=0.5)
        axs[i].set_aspect('equal')
        axs[i].set_xlim(-0.5, 0.5)
        axs[i].set_xlabel(r'$x_1$', fontsize=16)
        axs[i].set_ylabel(r'$x_2$', fontsize=16)
        axs[i].set_title(f'{sample_name}', fontsize=16)
        axs[i].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig('../Plots/QMC_vs_MC.eps', format='eps', dpi=300)
    plt.show()

def sample_and_plot(domain, s, y, nr_realizations=1):
    domain_x, domain_y = domain.points

    # Dw = get_V_affine(domain_x, domain_y, y, s)
    Dw = get_V_exp(domain_x, domain_y, y, s)
    # Dw = get_V_Chernov(domain_x, domain_y, y, s)
    DwX = Dw[0]
    DwY = Dw[1]

    plt.figure(figsize=(12, 7))

    plt.subplot(1, 2, 1)
    x_boundary, y_boundary = domain.get_boundary()
    plt.plot(x_boundary, y_boundary)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    # plt.title('$D_{ref}$', fontsize=16)
    plt.axis('equal')
    plt.xticks(([-2, -1, 0, 1, 2]), fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(1, 2, 2)
    Dw_boundary = get_V_exp(x_boundary, y_boundary, y, s)
    # Dw_boundary = get_V_Chernov(x_boundary, y_boundary, y, s)
    plt.plot(Dw_boundary[0], Dw_boundary[1])

    # add D- and D+
    # Dw_min = get_V_exp_min(x_boundary, y_boundary, s)
    # plt.plot(Dw_min[0], Dw_min[1], '--', color='gray', label="$D^{-}$")
    # Dw_max = get_V_exp_max(x_boundary, y_boundary, s)
    # plt.plot(Dw_max[0], Dw_max[1], '--', color='black', label="$D^{+}$")

    for i in range(nr_realizations-1):
        y = np.random.uniform(-0.5, 0.5, size=s)
        Dw_boundary = get_V_exp(x_boundary, y_boundary, y, s)
        # Dw_boundary = get_V_Chernov(x_boundary, y_boundary, y, s)
        plt.plot(Dw_boundary[0], Dw_boundary[1])

    # theta = np.linspace(0, 2*np.pi, 100)
    # plt.plot(0.25*np.cos(theta), 0.25*np.sin(theta), color='red', label='$D_- = B_{1/4}((0,0))$')
    # disk = Circle((0, 0), 0.25, color='lightcoral')
    # plt.gca().add_patch(disk)

    plt.xlabel('$x_1$', fontsize=16)
    # plt.title(f'1000 realizations of $D(y)$', fontsize=16)
    # plt.legend(fontsize=16, borderpad=1)
    plt.axis('equal')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)

    # plt.savefig("../Plots/Realizations_sampled_domain_reconstruction.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.show()

    return DwX, DwY

def plot_2D_solution(uh, V):
    # Visualize the solution
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()

def plot_3D_solution(uh, V):
    # Plot solution using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract points and values for plotting
    points = V.tabulate_dof_coordinates().reshape((-1, 2))
    u_values = uh.vector().get_local()

    ax.plot_trisurf(points[:, 0], points[:, 1], u_values, cmap='viridis')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    plt.title(r"Solution to $\Delta u = f$ in $D(\omega)$")
    plt.savefig("../Plots/3D_solution_Poisson_eq.eps", format='eps', dpi=300, bbox_inches='tight')

    plt.show()

def mesh_and_solution(domain):
    s = 100

    y = np.random.uniform(-0.5, 0.5, size=s)
    DwX, DwY = sample(domain, s, y)
    mesh = FEM.create_mesh(DwX, DwY)
    uh, V = FEM.solve_poisson(mesh)

    points = mesh.coordinates()
    triangles = mesh.cells()

    fig = plt.figure(figsize=(12, 6))
    
    # First subplot - 2D mesh visualization
    axs0 = fig.add_subplot(121)
    axs0.triplot(points[:, 0], points[:, 1], triangles, color='black')
    axs0.set_xlabel("$x_1$", fontsize=16, labelpad=10)
    axs0.set_ylabel("$x_2$", fontsize=16, labelpad=10)
    axs0.tick_params(axis='both', which='major', labelsize=14)
    axs0.set_aspect('equal')

    # Second subplot - 3D plot of the solution
    axs1 = fig.add_subplot(122, projection='3d')
    points = V.tabulate_dof_coordinates().reshape((-1, 2))
    u_values = uh.vector().get_local()

    axs1.plot_trisurf(points[:, 0], points[:, 1], u_values, cmap='viridis')
    axs1.set_xlabel("$x_1$", fontsize=16, labelpad=10)
    axs1.set_ylabel("$x_2$", fontsize=16, labelpad=10)
    axs1.tick_params(axis='both', which='major', labelsize=14)
    axs1.set_zlabel("$u(x,y_s)$", fontsize=16, labelpad=10)

    plt.subplots_adjust(wspace=0.05)  

    plt.savefig("../Plots/mesh_and_solution.eps", format='eps', dpi=300, transparent=True)
    plt.show()

def domain_deformation():
    y = np.random.uniform(-0.5, 0.5, 50)
    def V_wrapper(x1, x2, y):
        Dw = get_V_affine(x1, x2, y, 50)
        DwX = Dw[0]
        DwY = Dw[1]
        old_modulus = np.sqrt(x1**2 + x2**2)
        new_modulus = np.sqrt(DwX**2 + DwY**2)
        return new_modulus - old_modulus
    
    def is_in_domain(x1, x2):
        return x1**2 + x2**2 <= 1

    def deformation_product(x1,x2,y):
        return V_wrapper(x1,x2,y)*is_in_domain(x1,x2)


    x1 = np.linspace(-1.5, 1.5, 1000)
    x2 = np.linspace(-1.5, 1.5, 1000)
    theta = np.linspace(0, 2*np.pi, 100)
    X, Y = np.meshgrid(x1, x2)
    Z = deformation_product(X,Y,y)
    Z_masked = np.ma.masked_where(Z <= 0, Z)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Z_masked,extent=[-1.5, 1.5, -1.5, 1.5], cmap='plasma')
    plt.plot(np.cos(theta), np.sin(theta), color='black')
    plt.colorbar(label='Change in |V|')
    plt.axis('equal')
    
    Dw_boundary = get_V_affine(np.cos(theta), np.sin(theta), y, 50)
    plt.subplot(1, 2, 2)
    plt.plot(Dw_boundary[0], Dw_boundary[1])
    plt.axis('equal')

    plt.show()

def plot_mesh(mesh):
    coordinates = mesh.coordinates()
    triangles = mesh.cells()

    plt.figure(figsize=(8, 6))
    plt.triplot(coordinates[:, 0], coordinates[:, 1], triangles, color='black')
    plt.title("Mesh Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.savefig("../Plots/mesh.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.show()    

def plot_ref_domain():
    def plot_polygon(polygon, label):
        x, y = polygon.exterior.xy
        plt.plot(x, y, label=label)

    num_points = 100

    # Create instances
    ellipse = Ellipse(num_points)
    
    # Plot original and perturbed boundaries for Ellipse
    plt.subplot(1, 2, 2)
    x, y = ellipse.get_boundary()
    plt.plot(x, y, label='Original Ellipse')
    plot_polygon(ellipse.Dmin, 'Perturbed Ellipse')
    plt.legend()
    plt.title('Ellipse')

    plt.show()

def triple_plot(domain, s, y):
    DwX, DwY = sample(domain, s, y)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    x_boundary, y_boundary = domain.get_boundary()
    plt.plot(x_boundary, y_boundary)
    plt.ylabel('$x_2$', fontsize=16)
    plt.title('$D_{ref}$', fontsize=16)
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xticks(([-2, -1, 0, 1, 2]), fontsize=14)
    plt.yticks(([-2, -1, 0, 1, 2]), fontsize=14)

    plt.subplot(1, 3, 2)
    Dw_boundary = get_V_exp(x_boundary, y_boundary, y, s)
    plt.plot(Dw_boundary[0], Dw_boundary[1])

    plt.xlabel('$x_1$', fontsize=16)
    plt.title(f'$D(y)$, s = {s}', fontsize=16)
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xticks(([-2, -1, 0, 1, 2]), fontsize=14)
    plt.yticks(([-2, -1, 0, 1, 2]), fontsize=14)

    plt.subplot(1, 3, 3)

    mesh = create_mesh(DwX, DwY, max_angle=7, max_edge=0.25)
    coordinates = mesh.coordinates()
    triangles = mesh.cells()

    plt.triplot(coordinates[:, 0], coordinates[:, 1], triangles, color='black')
    plt.title(f'mesh for $D(y)$', fontsize=16)
    plt.axis('equal')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xticks(([-2, -1, 0, 1, 2]), fontsize=14)
    plt.yticks(([-2, -1, 0, 1, 2]), fontsize=14)

    plt.savefig("../Plots/Ellipse_sample_mesh.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.show()

    return DwX, DwY

def show_point_evaluations(domain, s):
    og_eval_xs, og_eval_ys = domain.evaluation_points
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for i, ax in enumerate(axes):
        y = np.random.uniform(-0.5, 0.5, size=s)
        DwX, DwY = sample(domain, s, y)
        eval_xs, eval_ys = get_V_exp(og_eval_xs, og_eval_ys, y, s)
        x_boundary, y_boundary = domain.get_boundary()
        Dw_boundary_x, Dw_boundary_y = get_V_exp(x_boundary, y_boundary, y, s)

        ax.scatter(og_eval_xs, og_eval_ys, s=4, color='blue', label='evaluation points in $D_{ref}$')
        ax.scatter(eval_xs, eval_ys, s=4, color='coral', label='evaluation points in $D(y)$')
        ax.plot(x_boundary, y_boundary, color='blue')
        ax.plot(Dw_boundary_x, Dw_boundary_y, color='coral')

        # Labels and adjustments
        if i == 0:
            ax.set_ylabel('$x_2$', fontsize=16)
        if i == 1:
            ax.set_xlabel('$x_1$', fontsize=16)

        ax.axis('equal')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.tick_params(axis='x', labelsize=16)  # Set x-tick label size
        ax.tick_params(axis='y', labelsize=16)  # Set y-tick label size

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    fig.savefig("../Plots/Points_evaluations.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.show()

def plot_mesh_only(domain):
    s = 100

    y = np.random.uniform(-0.5, 0.5, size=s)
    DwX, DwY = sample(domain, s, y)
    mesh = FEM.create_mesh(DwX, DwY)

    points = mesh.coordinates()
    triangles = mesh.cells()

    fig, ax = plt.subplots(figsize=(8, 8))  # Create a single plot
    ax.triplot(points[:, 0], points[:, 1], triangles, color='black')
    ax.set_xlabel("$x_1$", fontsize=18, labelpad=10)
    ax.set_ylabel("$x_2$", fontsize=18, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_aspect('equal')

    fig.savefig("../Plots/Mesh.eps", format='eps', dpi=300, bbox_inches='tight', transparent=True)

    plt.show()

if __name__ == '__main__':
    s = 100
    y = np.random.uniform(-0.5, 0.5, size=s)
    # sample_and_plot(Ellipse(num_points=128, a = 1, b = 1), s, y, nr_realizations=1000)
    # show_single_point_evaluations(Ellipse(num_points=128, a = 1, b = 1), s)
    # sample_and_plot(Torus(num_points=128, a_max = 2, b_max = 1, a_min = 1, b_min = 0.5), s, y, nr_realizations=1)
    # mesh_and_solution(Ellipse(num_points=128, a = 1, b = 1))
    # sample_and_plot(Hexagon(num_points=200, l=1), s, y)
    # triple_plot(Ellipse(num_points=80, a=1, b=1), s, y)
    # triple_plot(Ellipse(num_points=50, a=1, b=0.5), s, y)
    # qmc_vs_mc()
    plot_mesh_only(Ellipse(num_points=90, a = 1, b = 1))