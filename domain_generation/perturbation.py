from numpy import arange, tensordot, newaxis, cos, pi, arctan2, exp

def get_V_affine(x1, x2, y, s):
    i_values = arange(1, s + 1)

    phi = (1 / i_values[:, newaxis, newaxis]**2) * cos(i_values[:, newaxis, newaxis] * pi * arctan2(x1, x2)/pi)
    V = 1 + tensordot(y, phi, axes=(0, 0))

    V1 = V * x1
    V2 = V * x2

    V = [V1[0], V2[0]]

    return V

def get_V_exp(x1, x2, y, s):
    i_values = arange(1, s + 1)

    phi = (1 / i_values[:, newaxis, newaxis]**2.1) * 1.2 * cos(3 * i_values[:, newaxis, newaxis] * arctan2(x1, x2) - pi/2)
    xi = exp(-1/(y+0.5))
    V = 1 + tensordot(xi, phi, axes=(0, 0))

    V1 = V * x1
    V2 = V * x2

    V = [V1[0], V2[0]]

    return V

def get_V_Chernov(x1, x2, y, s):
    i_values = arange(1, s + 1)

    phi = (1 / i_values[:, newaxis, newaxis]**2.1) * 3 * cos(i_values[:, newaxis, newaxis] * x1) * cos(i_values[:, newaxis, newaxis]**2 * x2)
    xi = exp(-1/(y+0.5))
    V = 1 + tensordot(xi, phi, axes=(0, 0))

    V1 = V * x1
    V2 = V * x2

    V = [V1[0], V2[0]]

    return V

def reconstruct_V_exp(x1, x2, y, s):
    # Since from Experiment.get_results we get \xi(y), we cannot use get_V_exp straight away to recover the field V

    i_values = arange(1, s + 1)

    phi = (1 / i_values[:, newaxis, newaxis]**2.1) * 1.2 * cos(3 * i_values[:, newaxis, newaxis] * arctan2(x1, x2) - pi/2)
    xi = y
    V = 1 + tensordot(xi, phi, axes=(0, 0))

    V1 = V * x1
    V2 = V * x2

    V = [V1[0], V2[0]]

    return V