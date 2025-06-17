from .perturbation import *

'''
Calls a given perturbation field to generate the sampled domain.
Even though the implementation seems redundant, it makes for a nicer flow in other modules.
'''

def sample(domain, s, y=None):
    domain_x, domain_y = domain.points
    # Dw = get_V(domain_x, domain_y, y, s)
    Dw = get_V_exp(domain_x, domain_y, y, s)
    DwX = Dw[0]
    DwY = Dw[1]

    return DwX, DwY
