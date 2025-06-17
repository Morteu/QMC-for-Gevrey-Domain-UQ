from numpy import linspace, meshgrid, cos, sin, pi, array, append, concatenate, vstack
from shapely.geometry import Polygon, Point

'''
Implements the Domain class for a Disk, a Square, and a Hexagon (all in 2D)
'''

class Domain:
    def __init__(self, num_points):
        self.num_points = num_points
        self.grid = self.generate_grid()
        self.points = self.get_points()

    def generate_grid(self):
        x = linspace(-2, 2, self.num_points)
        y = x
        X, Y = meshgrid(x, y)
        return X, Y

    def get_boundary(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_points(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get_evaluation_points(self):
        raise NotImplementedError("Subclass must implement abstract method")

class Ellipse(Domain):
    def __init__(self, num_points, a=2, b=1):
        self.a = a
        self.b = b
        super().__init__(num_points)
        self.evaluation_points = self.get_evaluation_points()
        self.name = 'Ellipse' 

    def get_boundary(self):
        theta = linspace(0, 2 * pi, self.num_points)
        x = self.a * cos(theta)
        y = self.b * sin(theta)
        return x, y

    def get_points(self):
        X, Y = self.grid
        mask = (X / self.a)**2 + (Y / self.b)**2 <= 1
        return X[mask], Y[mask]
    
    def get_evaluation_points(self, n_eval_points=5, n_levels = 2):
        rho = 2*(n_eval_points - 1)/(self.a*(n_levels-1)) #FIXME so far this only works as intended for a disk, a=b

        rs = linspace(0, self.a ,n_levels+1)[1:-1]
        xs = array([0])
        ys = array([0])    
        for r in rs:
            n_theta = int(r*rho)
            thetas = linspace(0, 2*pi, n_theta, endpoint=False) + pi/4
            new_xs, new_ys = r*cos(thetas), r*sin(thetas)

            xs = append(xs, new_xs)
            ys = append(ys, new_ys)

        return xs, ys
    
class Square(Domain):
    def __init__(self, num_points, a=1):
        self.a = a
        super().__init__(num_points)
        self.name = 'Square'

    def get_boundary(self): # We don't need more resolution in the boundary since we will not perturbe the square
        half_side = self.a / 2
        x = array([-half_side, half_side, half_side, -half_side, -half_side])
        y = array([-half_side, -half_side, half_side, half_side, -half_side])
        return x, y

    def get_points(self):
        X, Y = self.grid
        half_side = self.a / 2
        mask = (abs(X) <= half_side) & (abs(Y) <= half_side)
        return X[mask], Y[mask]

class Hexagon(Domain):
    def __init__(self, num_points, l=1):
        self.l = l  # side length of the hexagon
        super().__init__(num_points)
        self.name = 'Hexagon'

    def get_boundary(self):
        # Generate the vertices of a regular hexagon
        angles = linspace(0, 2 * pi, 7)[:-1]  # 6 vertices, exclude the last 2*pi
        vertices_x = self.l * cos(angles)
        vertices_y = self.l * sin(angles)

        # Generate points along the edges of the hexagon
        edge_points_x = []
        edge_points_y = []
        
        # Loop over each pair of consecutive vertices and interpolate between them
        for i in range(len(vertices_x)):
            x_start, y_start = vertices_x[i], vertices_y[i]
            x_end, y_end = vertices_x[(i + 1) % len(vertices_x)], vertices_y[(i + 1) % len(vertices_y)]
            
            # Interpolate along the edge between these two vertices
            num_edge_points = max(2, self.num_points // 6)
            t = linspace(0, 1, num_edge_points)
            x_edge = x_start + t * (x_end - x_start)
            y_edge = y_start + t * (y_end - y_start)
            
            edge_points_x.extend(x_edge)
            edge_points_y.extend(y_edge)

        return array(edge_points_x), array(edge_points_y)

    def get_points(self):
        # Generate the hexagon vertices
        angles = linspace(0, 2 * pi, 7)[:-1]  # 6 vertices
        vertices_x = self.l * cos(angles)
        vertices_y = self.l * sin(angles)

        # Create a Polygon object using the vertices
        hexagon_polygon = Polygon(zip(vertices_x, vertices_y))

        # Generate the grid of points (X, Y) that encompasses the hexagon
        X, Y = self.grid  # Assuming `self.grid` is a meshgrid created in `Domain`

        # Flatten the grid arrays for easier processing
        points = vstack([X.ravel(), Y.ravel()]).T

        # Check which points are inside the polygon
        mask = array([hexagon_polygon.contains(Point(x, y)) for x, y in points])

        # Reshape the mask to the original grid shape and return the points inside
        return X.ravel()[mask], Y.ravel()[mask]
    

if __name__ == '__main__':
    domain = Ellipse(256, a=1, b=1)
    xs, ys = domain.evaluation_points
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,7))
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.scatter(xs, ys, s=1, color='red')
    plt.title('Point evaluation in $D_-$, $n_p$ =' + f'{len(xs)}', fontsize=16)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)

    plt.show()