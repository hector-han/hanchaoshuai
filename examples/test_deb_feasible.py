import numpy as np
import matplotlib.pyplot as plt
from utils import good_point_init, deb_feasible_compare

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def obj_fun(x):
    return x[0] ** 2 + x[1] ** 2


def less_1(x):
    return 0.5 - x[0]

def less_2(x):
    return x[0] - 1.0

def less_3(x):
    return 0.5 - x[1]

def less_4(x):
    return x[1] - 1.0


if __name__ == '__main__':
    # 随机生成的点
    num_of_points = 50
    good_points = good_point_init(num_of_points, np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]))

    i = deb_feasible_compare(good_points, obj_fun=obj_fun, equ_cons=[], less_cons=[less_1, less_2, less_3, less_4])
    print(i)
    print(good_points[i])