from numpy import random, mod, apply_along_axis, dot, exp, arange, sqrt, pi, ones, minimum, zeros, vstack, array, real, argmin
from inverse import *
from scipy.fftpack import fft, ifft
from scipy.special import zeta, factorial
from pandas import DataFrame, read_csv

class Experiment:
    def __init__(self, Dref, s, delta, gamma):
        self.Dref = Dref
        self.s = s
        self.delta = delta
        self.gamma = gamma

    def get_results(self, ns, q):
        column_names = ['Method', 'n', 'shift_num'] + [f'new_y_{i}' for i in range(self.s)]
        results_df = DataFrame(columns=column_names)
        z_gen = load_generating_vector("inputs/gen_vector.csv", self.s)

        for n in ns:
            z = fastcbc(n, self.s)
            tailored_qmc_ys = get_samples(z, n)
            generic_qmc_ys = get_samples(z_gen, n)
            mc_ys = random.uniform(0, 1, size=(n,self.s))
            # yss = [('Weight-dependent QMC',tailored_qmc_ys), ('MC',mc_ys), ('Generic QMC',generic_qmc_ys)]
            yss = [('Weight-dependent QMC',tailored_qmc_ys), ('MC',mc_ys)]

            for ys in yss: 
                for k in range(q):
                    shift = random.uniform(0, 1, (1, self.s))
                    shifted_ys = mod(ys[1] + shift, 1) - 0.5

                    nu_deltas = apply_along_axis(lambda row: get_EV_integrands(self.Dref, self.s, row, self.delta, self.gamma), 1, shifted_ys)
                    nu_delta_ys = dot(nu_deltas, exp(-1/(shifted_ys+0.5)))

                    new_y = nu_delta_ys / nu_deltas.sum()
                    results_df.loc[len(results_df)] = [ys[0], n, k+1] + new_y.flatten().tolist()

            results_df.to_csv(f'Results/results_n{n}.csv', index=False)

def bernoulli(x):
    return x**2 - x + 1/6

def generator(n):
    """For prime n, find the primitive root modulo n."""

    def is_prime(n):
        """Simple check to determine if n is a prime number."""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def factor(n):
        """Returns the list of factors of n."""
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors
    
    if not is_prime(n):
        raise ValueError('n is not a prime')

    factorlist = list(set(factor(n - 1)))
    g = 2
    i = 0
    while i < len(factorlist):
        if pow(g, (n - 1) // factorlist[i], n) == 1:
            g += 1
            i = 0
        else:
            i += 1
    return g

def fastcbc(n, s):
    """
    Fast CBC construction for generating a sequence of quasi-random points.
    """
    m = (n - 1) // 2
    
    C_nu = 1
    beta = 2
    b = C_nu * arange(1, s + 1, dtype=float)**(-2.1)
    delta = 0.05
    lambda_val = 1 / (2 - 2 * delta)
    Gammaratio = arange(2, s + 2)**(2*beta / (1 + lambda_val))
    gamma = (b / sqrt(2 * zeta(2 * lambda_val) / (2 * pi**2)**lambda_val))**(2 / (1 + lambda_val))

    # Rader permutation
    g = generator(n)
    perm = ones(m, dtype=int)
    for j in range(1, m):
        perm[j] = mod(perm[j-1] * g, n)
    perm = minimum(perm, n - perm)

    # Precompute the FFT of the Bernoulli polynomial evaluated at permuted indices
    omega_input = mod(perm * perm[-1] / n, 1)
    fftomega = fft(bernoulli(omega_input))
    
    z = []
    p = zeros((s, m))

    for d in range(s):
        pold = vstack([ones(m), p])
        x = gamma[d] * Gammaratio @ pold[:-1, :]
        
        if d == 0:
            minind = 1

        else:
            flipped_perm = perm[::-1]
            tmp = real(ifft(fftomega * fft(x[flipped_perm - 1])))
            minind_idx = argmin(tmp)
            minind = perm[minind_idx]
        
        z.append(minind)
        omega = bernoulli(mod(minind * arange(1, m + 1) / n, 1))
        
        for l in range(d + 1):
            p[l, :] = pold[l + 1, :] + omega * pold[l, :] * Gammaratio[l] * gamma[d]
    
    return array(z)

def get_samples(z, n):
    '''Generates n points from a generating vector z'''
    
    s = len(z)
    points = zeros((n, s))
    for i in range(n):
        points[i, :] = mod((i+1) * z / n, 1)
    return points

def load_generating_vector(csv_file, s):
    '''Takes any given generating vector in a .csv and saves it to an array'''
    
    df = read_csv(csv_file, header=None)
    data = df[0].to_numpy()
    z = data[0:s]
    return z

if __name__ == "__main__":
    s = 100
    n = 127
    z = fastcbc(n, s)
    print(z)
    tailored_qmc_ys = get_samples(z, n)

    # print(regularity_constants(sigma_min=0.1, sigma_max=3, C_poin=3.4, C=2, beta=2, u_H10=0.74, u_inf=0.98, tau_min=0.2, Dref_area=1, common_domain_area=0.3, rho_norm=1, d=2))
