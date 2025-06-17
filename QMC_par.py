import numpy as np
from inverse import *
from scipy.fftpack import fft, ifft
from scipy.special import zeta
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

'''
This module implements the Experiment class with 3 threads (one per integration method). 
It should be combined with export OMP_NUM_THREADS=n_cpus/3 to make sure that all methods
run in a similar time.
'''

class Experiment:
    def __init__(self, Dref, s, delta, gamma):
        self.Dref = Dref
        self.s = s
        self.delta = delta
        self.gamma = gamma

    def process_ys(self, ys, n, q):
        """Process the shift operations and return the results for each ys."""
        results = []
        for k in range(q):
            shift = np.random.uniform(0, 1, (1, self.s))
            shifted_ys = np.mod(ys[1] + shift, 1) - 0.5

            nu_deltas = np.apply_along_axis(lambda row: get_EV_integrands(self.Dref, self.s, row, self.delta, self.gamma), 1, shifted_ys)
            nu_delta_ys = np.dot(nu_deltas, np.exp(-1/(shifted_ys+0.5)))

            new_y = nu_delta_ys / nu_deltas.sum()
            results.append([ys[0], n, k+1] + new_y.flatten().tolist())

        return results

    def get_results(self, folder, ns, q):
        column_names = ['Method', 'n', 'shift_num'] + [f'new_y_{i}' for i in range(self.s)]
        results_df = pd.DataFrame(columns=column_names)
        z_gen = load_generating_vector("inputs/gen_vector.csv", self.s)

        for n in ns:
            z = fastcbc(n, self.s)
            tailored_qmc_ys = get_samples(z, n)
            generic_qmc_ys = get_samples(z_gen, n)
            mc_ys = np.random.uniform(0, 1, size=(n, self.s))
            yss = [('Weight-dependent QMC', tailored_qmc_ys), ('MC', mc_ys), ('Generic QMC', generic_qmc_ys)]

            # Use ProcessPoolExecutor to process each ys in parallel
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.process_ys, ys, n, q) for ys in yss]

                # TODO: try if parallelizing on the shifts (for each method) instead of the methods yields better results

                for future in futures:
                    results = future.result()  # Get the result from each parallel process
                    for result in results:
                        results_df.loc[len(results_df)] = result

            # Save results to CSV after each n
            path = folder + f'/results_par_n{n}.csv'
            results_df.to_csv(path, index=False)


def bernoulli(x):
    return x**2 - x + 1/6

def generator(n):
    """For prime n, find the primitive root modulo n."""
    def is_prime(n):
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
    m = (n - 1) // 2
    C_nu = 1
    beta = 2
    b = C_nu * np.arange(1, s + 1, dtype=float)**(-2.1) 
    delta = 0.05
    lambda_val = 1 / (2 - 2 * delta)
    Gammaratio = np.arange(2, s + 2)**(2*beta / (1 + lambda_val))
    gamma = (b / np.sqrt(2 * zeta(2 * lambda_val) / (2 * np.pi**2)**lambda_val))**(2 / (1 + lambda_val))

    g = generator(n)
    perm = np.ones(m, dtype=int)
    for j in range(1, m):
        perm[j] = np.mod(perm[j-1] * g, n)
    perm = np.minimum(perm, n - perm)

    omega_input = np.mod(perm * perm[-1] / n, 1)
    fftomega = fft(bernoulli(omega_input))
    
    z = []
    p = np.zeros((s, m))

    for d in range(s):
        pold = np.vstack([np.ones(m), p])
        x = gamma[d] * Gammaratio @ pold[:-1, :]
        
        if d == 0:
            minind = 1
        else:
            flipped_perm = perm[::-1]
            tmp = np.real(ifft(fftomega * fft(x[flipped_perm - 1])))
            minind_idx = np.argmin(tmp)
            minind = perm[minind_idx]
        
        z.append(minind)
        omega = bernoulli(np.mod(minind * np.arange(1, m + 1) / n, 1))
        
        for l in range(d + 1):
            p[l, :] = pold[l + 1, :] + omega * pold[l, :] * Gammaratio[l] * gamma[d]
    
    return np.array(z)

def get_samples(z, n):
    s = len(z)
    points = np.zeros((n, s))
    for i in range(n):
        points[i, :] = np.mod((i + 1) * z / n, 1)
    return points

def load_generating_vector(csv_file, s):
    df = pd.read_csv(csv_file, header=None)
    data = df[0].to_numpy()
    z = data[0:s]
    return z

if __name__ == "__main__":
    s = 20
    n = 167
    z = fastcbc(n, s)
    print(z)
    tailored_qmc_ys = get_samples(z, n)
    # print(tailored_qmc_ys)
