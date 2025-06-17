from datetime import datetime

def log_parameters(folder, s, domain, og_domain, og_s, noise_level, q):
    """
    Outputs a .txt file with the relevant parameters for a given experiment.
    
    Args:
        folder (str): Path to save the file.
        s (int): Stochastic dimension used for the experiment.
        domain (Domain): Reference domain.
        og_domain (Domain): Domain used to generate the data.
        og_s (int): Stochastic dimension used to generate og_domain.
        noise_level (float): Noise level for the covariance of additive noise \eta.
        q (int): Number of random shifts used in the QMC approximation.

    Returns: 
        None
    """

    path = folder + '/parameters.txt'
    num_points = domain.num_points/4
    og_num_points = og_domain.num_points/4
    ref_domain_name = domain.name
    og_domain_name = og_domain.name
    n_eval_points = len(domain.evaluation_points[0])

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(path, 'w') as file:
        file.write(f'File saved on {current_time}\n\n')
        file.write(f'Reference domain: {ref_domain_name}\n')
        file.write(f's_ref = {s}\n')
        file.write(f'h = 1/{int(num_points)}\n\n')

        file.write(f'Domain used as measurement model: {og_domain_name}\n')
        file.write(f's_og = {og_s}\n')
        file.write(f'h_og = 1/{int(og_num_points)}\n\n')

        file.write(f'noise level = {noise_level*100}%\n')
        file.write(f'number of shifts: {q}\n')
        file.write(f'number of evaluation points: {n_eval_points}\n')