from scipy import integrate
import numpy as np
from math import pow, factorial

def cal_integerate(t_0, lam, m):
    """
    计算积分
    :param t_0:
    :param lam:
    :param m:
    :return:
    """
    def _inner_func(t):
        part1 = np.exp(-lam * t)
        part2 = np.sum([pow(lam * t, j) / factorial(j) for j in range(m + 1)])
        return part1 * part2
    result = integrate.quad(_inner_func, 0, t_0)
    print(result)
    return result[0]

if __name__ == '__main__':
    t_0 = 540
    tmp = cal_integerate(t_0, 1e-4, 10)
    t_x = t_0 * (1 - tmp / t_0)

    tmp = cal_integerate(t_x, 1e-3, 10)
    print(tmp, t_x)
