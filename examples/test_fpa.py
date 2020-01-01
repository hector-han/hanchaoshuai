import numpy as np
import math
from fpa import FlowerPollinationAlgorithm

def obj_fun(x):
    x1 = x[0]
    x2 = x[1]
    _sqrt = math.sqrt(x1**2+x2**2)
    # ret = math.sin(x1) * math.exp((1-math.cos(x2))**2) + math.cos(x2) * math.exp((1-math.sin(x1))**2) + (x1 - x2) **2
    ret = -math.sin(_sqrt) / _sqrt -math.exp(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2) / 2) + 0.7129
    return np.asarray([ret])


def transformer(v):
    return v


lb = np.asarray([-10, -10])
ub = -lb

if __name__ == '__main__':
    fpa = FlowerPollinationAlgorithm(2, obj_fun, transformer, N_iter=100, lb=lb, ub=ub, name='bridge')
    fpa.train()
    fpa.save('../output', axis=[0, 1], axis_name=['x', 'y'], f_and_cons_name=['f_val'])