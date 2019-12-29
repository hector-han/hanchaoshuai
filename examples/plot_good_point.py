import numpy as np
import matplotlib.pyplot as plt
from utils import good_point_init, random_point_init

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 随机生成的点
    num_of_points = 100
    lb = np.asarray([-2, 2])
    ub = np.asarray([-1, 3])
    random_points = random_point_init(num_of_points, lb, ub)
    good_points = good_point_init(num_of_points, lb, ub)

    plt.subplot(121)
    plt.scatter(random_points[:, 0], random_points[:, 1])
    plt.title('随机生成的种群A')

    plt.subplot(122)
    plt.scatter(good_points[:, 0], good_points[:, 1])
    plt.title('基于佳点集生成的种群B')

    plt.show()