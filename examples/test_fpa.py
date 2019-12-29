import numpy as np
import math
from fpa import FlowerPollinationAlgorithm

def obj_fun(x):
    x1 = x[0]
    x2 = x[1]
    ret = math.sin(x1) * math.exp((1-math.cos(x2))**2) + math.cos(x2) * math.exp((1-math.sin(x1))**2) + (x1 - x2) **2
    return ret

lb = np.asarray([-2*math.pi, -2*math.pi])
ub = -lb

if __name__ == '__main__':
    fpa = FlowerPollinationAlgorithm(obj_fun, 2, N_iter=100, lb=lb, ub=ub, name='brid')
    fpa.train()
    fpa.save('../output')