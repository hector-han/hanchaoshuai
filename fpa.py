import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from utils import good_point_init, random_point_init, deb_feasible_compare
from math import gamma, sin, pi, pow

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                    datefmt='%Y-%m-%d %A %H:%M:%S')

"""
 Flower pollenation algorithm (FPA), or flower algorithm
 花朵授粉算法
 
"""

class FlowerPollinationAlgorithm(object):
    def __init__(self, obj_fun, n_dim, less_cons=[], lb=None, ub=None, num_popu=100, N_iter=1000,
                 p=0.8, int_method='good', integer_op=False, coef=0.01, name='fpa'):
        """
        花朵授粉算法
        :param obj_fun: 目标函数，极小化这个函数
        :param n_dim: 自变量维度
        :param less_cons: 不等式约束列表，f(x) <= 0 for f in less_cons
        :param lb: 自变量下界约束
        :param ub: 自变量上界约束
        :param num_popu: 初始种群个数
        :param N_iter: 迭代次数
        :param p: 自花授粉概率， 1-p为全局授粉概率
        :param int_method: 初始化方式，good: 佳点集方式，否则随机
        :param integer_op: 是否是整数规划，默认False
        :param name: 模型名字，保存图片会使用这个作为前缀
        """
        self.obj_fun = obj_fun
        self.n_dim = n_dim
        self.less_cons = less_cons
        self.lb = lb
        self.ub = ub
        self.num_popu = num_popu
        self.N_iter = N_iter
        self.p = p
        self.integer_op = integer_op
        self.coef = coef
        self.name = name

        if self.lb is not None:
            init_lb = self.lb
        else:
            init_lb = -1000 * np.ones(n_dim)
        if self.ub is not None:
            init_ub = self.ub
        else:
            init_ub = 1000 * np.ones(n_dim)

        if int_method == 'good':
            self.populations = good_point_init(num_popu, init_lb, init_ub)
        else:
            self.populations = random_point_init(num_popu, init_lb, init_ub)

        if self.integer_op:
            self.populations = self.populations.astype(np.int64)

        self.f_min_list = []
        self.x_best_list = []
        self.diversity_list = []

    def _diversity(self):
        center = np.mean(self.populations, axis=0)
        tmp = self.populations - center
        part_norm = np.linalg.norm(tmp, axis=1)
        return np.sqrt(np.sum(part_norm ** 2)) / self.num_popu / np.max(part_norm)

    def levy(self):
        beta = 3 / 2
        sigma = (gamma(1+beta)*sin(pi*beta/2) /
                 (gamma((1+beta)/2) * beta * (2 ** ((beta-1)/2)))
                 ) ** (1/beta)
        u = np.random.randn(self.n_dim) * sigma
        v = np.random.randn(self.n_dim)
        step = u / np.power(np.abs(v), 1 / beta)
        return self.coef * step

    def _bound(self, x):
        x_tmp = x
        if self.lb is not None:
            _mask = x_tmp < self.lb
            x_tmp[_mask] = self.lb[_mask]

        if self.ub is not None:
            _mask = x_tmp > self.ub
            x_tmp[_mask] = self.ub[_mask]

        if self.integer_op:
            x_tmp = x_tmp.astype(np.int64)
        return x_tmp

    def train(self):
        logging.info('fpa begin to train...')
        _flag = False
        idx, _ = deb_feasible_compare(self.populations, self.obj_fun, [], self.less_cons)
        if _:
            _flag = True
        x_best = self.populations[idx]
        self.x_best_list.append(x_best)
        f_min = self.obj_fun(x_best)

        # 开始按照t迭代
        print_steps = self.N_iter // 10
        for t in range(self.N_iter):
            self.f_min_list.append(f_min)
            self.diversity_list.append(self._diversity())
            if t % print_steps == 0:
                logging.info('t={}, f_min={}'.format(t, f_min))

            # 对每一个解迭代
            for i in range(self.num_popu):
                if np.random.random() > self.p:
                    # levy 飞行， 生物授粉 x_i^{t+1}=x_i^t+ L (x_i^t-gbest)
                    L = self.levy()
                    dS = L * (self.populations[i] - x_best)
                    x_new = self.populations[i] + dS
                    x_new = self._bound(x_new)
                else:
                    # 非生物授粉 x_i^{t+1}+epsilon*(x_j^t-x_k^t)
                    epsilon = np.random.random()
                    JK = np.random.permutation(self.num_popu)
                    x_new = self.populations[i] + epsilon * (self.populations[JK[0]] - self.populations[JK[1]])
                    x_new = self._bound(x_new)
                tmp = [self.populations[i], x_new]
                idx, _ = deb_feasible_compare(tmp, self.obj_fun, [], self.less_cons)
                if idx == 1:
                    # 新解更优
                    self.populations[i] = x_new
                    tmp = [x_best, x_new]
                    idx, _ = deb_feasible_compare(tmp, self.obj_fun, [], self.less_cons)
                    if _:
                        _flag = True
                    if idx == 1:
                        x_best = x_new
                        self.x_best_list.append(x_best)
                        f_min = self.obj_fun(x_best)

        logging.info('最终是否找到可行解{}, 最优函数值{}'.format(_flag, f_min))

    def save(self, path):
        plt.figure()
        plt.plot(self.f_min_list)
        plt.savefig(os.path.join(path, '{}_f_min.jpg'.format(self.name)))

        plt.figure()
        plt.plot(self.diversity_list)
        plt.savefig(os.path.join(path, '{}_diversity.jpg'.format(self.name)))

        plt.figure()
        tmp = np.asarray(self.x_best_list)
        plt.plot(tmp[:, 0], tmp[:, 1])
        plt.savefig(os.path.join(path, '{}_x_best.jpg'.format(self.name)))





