from numpy import linalg, degrees, vstack, arccos, unique, array
from scipy.spatial import Delaunay
from dolfin import set_log_level, LogLevel, Mesh, MeshEditor, FunctionSpace, DirichletBC, Constant, TrialFunction, TestFunction, SpatialCoordinate, Function, solve, cos, sin
from ufl import ds, dx, grad, inner
from functools import cache

set_log_level(LogLevel.WARNING)

def create_mesh(DwX, DwY, min_angle=10, max_edge = 0.1): #TODO: ideally max_edge should depend on the number of points n
    """
    Creates a mesh for the transported points (DwX, DwY)
    
    Args:
        DwX (array): x-coordinate of transported points.
        DwY (array): y-coordinate of transported points.
        min_angle (int): Minimum angle allowed in the mesh.
        max_edge (int): Maximum edge sized allowed in the mesh.

    Returns: 
        mesh (dolfin.mesh)
    """

    def filter_triangles(points, simplices):
        # TODO: throw error when too many triangles have been deleted
        def angles(simplex):
            p1, p2, p3 = points[simplex]
            a = linalg.norm(p2 - p3)
            b = linalg.norm(p1 - p3)
            c = linalg.norm(p1 - p2)
            angle1 = degrees(arccos((b**2 + c**2 - a**2) / (2 * b * c)))
            angle2 = degrees(arccos((a**2 + c**2 - b**2) / (2 * a * c)))
            angle3 = 180 - angle1 - angle2

            if a > max_edge or b > max_edge or c > max_edge:
                angle1 = 0

            return [angle1, angle2, angle3]

        valid_simplices = []
        valid_points_set = set()

        for simplex in simplices:
            if all(angle > min_angle for angle in angles(simplex)):
                valid_simplices.append(simplex)
                valid_points_set.update(simplex)

        return array(valid_simplices), valid_points_set

    # Stack the DwX and DwY to create points array and remove duplicates
    points = vstack((DwX, DwY)).T
    unique_points, unique_indices = unique(points, axis=0, return_index=True)

    if len(unique_points) < 3:
        raise ValueError("Not enough unique points to create a triangulation. Need at least 3 unique points.")

    try:
        tri = Delaunay(unique_points, incremental=True)
    except Exception as e:
        print("Delaunay triangulation failed.")
        print(f"Exception: {e}")
        return None

    valid_simplices, valid_points_set = filter_triangles(unique_points, tri.simplices)

    if len(valid_simplices) == 0:
        raise ValueError("No valid triangles found after filtering based on angles.")

    valid_points_list = sorted(valid_points_set)
    old_to_new_index = {old_index: new_index for new_index, old_index in enumerate(valid_points_list)}

    new_simplices = array([[old_to_new_index[idx] for idx in simplex] for simplex in valid_simplices])
    new_points = unique_points[valid_points_list]

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(len(new_points))
    editor.init_cells(len(new_simplices))
    
    for i, point in enumerate(new_points):
        editor.add_vertex(i, point)
    
    for i, simplex in enumerate(new_simplices):
        editor.add_cell(i, simplex)
    
    editor.close()
    
    return mesh

def solve_poisson(mesh):
    # Defining the finite element function space
    V = FunctionSpace(mesh, "Lagrange", 1)
    
    # Homogeneous boundary conditions
    def boundary(x, on_boundary):
        return on_boundary

    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # Define the variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    f = 10*sin(x[0]*x[1]) - 5*cos(x[0]+x[1])**2
    # f = (2*exp(3*(x[0]+x[1])**2) - 5*x[0]*x[1] - exp(x[0] - x[1]))
    
    # Bilinear and linear forms
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    
    # Create solution function
    uh = Function(V)
    
    # Solve the variational problem
    solve(a == L, uh, bc, solver_parameters={"linear_solver": "lu"})
    
    return uh, V