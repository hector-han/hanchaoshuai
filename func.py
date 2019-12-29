import numpy as np
from scipy.special import comb
from math import pow, factorial
from scipy import integrate
from utils import good_point_init


def cal_T_k(phi_k, t_b, t_w):
    """
    计算平均故障隔离时间
    :param phi_k: 故障隔离率
    :param t_b: bite平均故障隔离时间
    :param t_w: 人工平均故障隔离时间
    :return:
    """
    return np.multiply(phi_k, t_b) + np.multiply(1 - phi_k, t_w)

### 参数定义 ###
# 串联系统下标，从0开始
c_idx = [2, 3, 4, 6, 7]
# 并联系统下标
b_idx = [0, 1, 5]
# 备件供应保障周期内装备累计工作时间
t_0 = 540
# 系统平均任务持续工作时间
t_s = 6
# 装备再次出动时间
t_d = 2
# 待保障系统数
n_p = 1
# 器材仓库到现场平均临时周转时间
t_p1 = 80
# 供应站到现场平均临时周转时间
t_p2 = 500
# 单元故障率
lambda_cb = 1e-5 * np.asarray([120, 90, 30, 50, 50, 60, 40, 30])
# 单元拆装修复率
mu_cb = np.asarray([4.50, 4.80, 3.33, 3.33, 2.80, 4.00, 3.60, 3.50])
# 故障隔离率
phi_k = np.asarray([0.86765, 0.92548, 0.87524, 0.95647, 0.97365, 0.97248, 0.94361, 0.97548])
# bite平均故障隔离时间, 单位小时
t_b = np.asarray([5, 4, 4, 2, 1, 1, 2, 2]) / 60
# 人工平均故障隔离时间
t_w = np.asarray([0.10, 0.15, 0.05, 0.08, 0.10, 0.10, 0.15, 0.10])
# 现场更换失效模式频数比， 没找着哪里用了，先空着
beta = np.asarray([0.24, 0.20, 0.30, 0.43, 0.20, 0, 0.15, 0.12]) / 2
# bite 可隔离故障的更换模块数
bite = np.asarray([1, 1, 1, 1, 1, 0, 0, 0])
# 冗余数量
o_cb_bound = np.asarray([4, 3, 1, 1, 1, 3, 1, 1])
# 表决数量
k_cb = np.asarray([1, 1, 1, 1, 1, 1, 1, 1])
# 备件数量
m1_bound = np.asarray([2, 2, 1, 1, 1, 2, 1, 1])
m2_bound = np.asarray([3, 5, 2, 2, 1, 5, 3, 3])
m3_bound = np.asarray([10, 20, 15, 12, 15, 20, 5, 5])
# 质量
mass = np.asarray([20, 4, 2, 4, 3, 3, 2, 1])

### 以下参数是自动计算出来的，不需要手动赋值
# 平均故障隔离时间
T_k = cal_T_k(phi_k, t_b, t_w)
# 串联系统参数
lambda_c, phi_k_c, mu_c, T_K_c = lambda_cb[c_idx], phi_k[c_idx], mu_cb[c_idx], T_k[c_idx]
# 并联系统参数
lambda_b, phi_k_b, mu_b, T_K_b = lambda_cb[b_idx], phi_k[b_idx], mu_cb[b_idx], T_k[b_idx]
o_b_bound, k_b = o_cb_bound[b_idx], k_cb[b_idx]

### 设置参数结束


def cal_f_cb(t, lambda_cb):
    """
    计算失效概率
    :param t: 时间参数
    :param lambda_cb: 单元故障率
    :return:
    """
    return lambda_cb * np.exp(-lambda_cb * t)


def cal_D_t(R_t, phi_k, M_t):
    """
    任务可信度函数
    :param R_t:装备可靠度函数
    :param phi_k:故障检测个利率
    :param M_t: 发生故障后的维修度函数
    :return:
    """
    tmp = R_t + (1 - R_t) * phi_k * M_t
    return tmp


def cal_R_t(t, lambda_c_list, lambda_b_list, o_b_list, k_b_list):
    """
    t时刻执行任务的可靠度
    :param t: 时刻t
    :param lambda_c_list: 串联系统的单元的故障率
    :param lambda_b_list: 并联系统的单元的故障率
    :param o_b_list: 并联系统单元数量
    :param k_b_list: 并联系统表决模块数据
    :return:
    """
    nb = len(lambda_b_list)
    assert len(o_b_list) == nb
    assert len(k_b_list) == nb
    lambda_c_times_t = lambda_c_list * t
    lambda_b_times_t = lambda_b_list * t
    part1 = np.exp(-np.sum(lambda_c_times_t))

    part2_comps = np.zeros(nb)
    for j in range(1, nb + 1):
        _sum = 0
        for g in range(o_b_list[j-1] - k_b_list[j-1] + 1):
            tmp = comb(o_b_list[j-1], g) * np.power(1 - np.exp(-lambda_b_times_t[j-1]), g) * \
                  np.exp(-(o_b_list[j-1] - g) * lambda_b_times_t[j-1])
            _sum += tmp
        part2_comps[j-1] = _sum
    part2 = np.prod(part2_comps)

    return part1 * part2


def cal_M_t(t, mu_c_list, mu_b_list, o_b_list):
    """
    t时刻维修度
    :param t:
    :param mu_c_list: 拆装修复率
    :param mu_b_list:
    :param o_b_list:
    :return:
    """
    assert len(mu_b_list) == len(o_b_list)
    part1 = 1 - np.exp(-mu_c_list * t)
    part2 = np.power(1 - np.exp(-mu_b_list * t), o_b_list)
    return np.prod(part1) * np.prod(part2)


def _denominator(o_b, k_b):
    """
    一个辅助函数，在很多地方都用到了
    :param o_b:
    :param k_b:
    :return:
    """
    nb = len(o_b)
    denominator = np.zeros(nb)
    for j in range(1, nb + 1):
        tmp = factorial(k_b[j-1] - 1) * np.sum([1 / factorial(g) for g in range(1, o_b[j-1] + 1)]) + \
            np.sum([1 / (o_b[j-1] - g) for g in range(o_b[j-1] - k_b[j-1] + 1)])
        denominator[j-1] = tmp
    return denominator


def cal_phi_or_T_k(para_c, lambda_c, para_b, lambda_b, o_b, k_b):
    """
    计算故障隔离率 或 平均故障隔离时间，参数都是np.array
    phi_k需要在外边 1-结果
    :param para_c: 串联系统参数
    :param lambda_c:
    :param para_b:
    :param lambda_b: 并联系统参数
    :param o_b:
    :param k_b:
    :return:
    """
    denominator = _denominator(o_b, k_b)
    part1 = np.sum(np.multiply(para_c, lambda_c)) + np.sum(np.divide(np.multiply(para_b, lambda_b), denominator))
    part2 = np.sum(lambda_c) + np.sum(np.divide(lambda_b, denominator))

    return part1 / part2


def cal_MTBF_S(lambda_c, lambda_b, o_b, k_b):
    """平均故障间隔时间"""
    denominator = _denominator(o_b, k_b)
    tmp = np.sum(lambda_c) + np.sum(np.divide(lambda_b, denominator))
    return 1 / tmp


def cal_MTTR_S(mu_c, T_k_c, lambda_c, mu_b, T_K_b, lambda_b, o_b, k_b):
    """平均修复时间"""
    denominator = _denominator(o_b, k_b)
    part1 = np.sum((np.divide(1, mu_c) + T_k_c) * lambda_c) + np.sum(np.divide(np.multiply(np.divide(1, mu_b) + T_K_b, lambda_b),
                                                                  denominator))
    part2 = np.sum(lambda_c) + np.sum(np.divide(lambda_b, denominator))
    return part1 / part2


def cal_MMDT_S(t_d, f_c, mu_c, f_b, mu_b, o_b):
    """
    平均维修延误时间
    """
    one_minus_f_c = 1 - f_c
    one_minus_f_b = 1 - f_b
    log_c = np.log(np.divide(1, one_minus_f_c))
    log_b = np.log(np.divide(1, one_minus_f_b))
    part1 = (1 - np.prod(one_minus_f_b) * np.prod(one_minus_f_c)) / (np.sum(log_c) + np.sum(log_b))
    part2 = np.sum(log_c * np.exp(-mu_c * t_d) / mu_c)
    nb = len(o_b)
    part3_comp = np.zeros(nb)
    for i in range(1, nb + 1):
        tmp = np.sum([pow(-1, j+1) * comb(o_b[i-1], j) * np.exp(-j * mu_b[i-1] * t_d) / (j * mu_b[i-1])
                      for j in range(1, o_b[i-1] + 1)])
        part3_comp[i-1] = tmp
    part3 = np.sum(log_b * part3_comp)
    return part1 * (part2 + part3)


def cal_ht(ft):
    one_minus_ft = 1 - ft
    log_ft = np.log(1 / one_minus_ft)
    part1 = np.prod(one_minus_ft)
    part2 = np.sum(log_ft)
    return (1 - part1) / part2 * log_ft


def _cal_integerate(t_0, lam, m):
    """
    计算积分, integerate[ exp(-lam * t) * sum ((lam * t)^j / j!), 0, m), 0, t_0]
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
    return result[0]

    
def cal_MLDT_S(ht, t_0, lambda_cb, o_cb, m1, m2, m3, n_p, t_p1, t_p2):
    n_cb = len(lambda_cb)
    complex_part = np.zeros(n_cb)
    for i in range(1, n_cb + 1):
        # t_x = t_0 * (1 - _cal_integerate(t_0, lambda_cb[i-1], m1[i-1]) / t_0)
        _lamb = n_p * o_cb[i-1] * lambda_cb[i-1]
        tmp1 = _cal_integerate(t_0, _lamb, m1[i-1])
        # tmp2 = _cal_integerate(t_x, _lamb, m2[i-1])
        tmp3 = _cal_integerate(t_0, _lamb, m1[i-1] + m2[i-1])
        # tmp4 = _cal_integerate(t_x, _lamb, m3[i-1])
        # complex_part[i-1] = t_p1 * (1 - tmp1 / t_0) * (tmp2 / t_x) + \
        #                   t_p2 * (1 - tmp3 / t_0) * (tmp4 / t_x)
        complex_part[i-1] = t_p1 * (1 - tmp1 / t_0) + t_p2 * (1 - tmp3 / t_0)
    return np.sum(ht * complex_part)


def cal_MPMDT_S():
    return 0


def obj_fun(x):
    """
    定义目标函数
    :param x: 自变量 3+3*8 = 27维的向量，表示3个o_i, 8个m1_i, m2_i, m3_i
    :return: 目标函数值
    """
    nb = 3
    o_b = x[0:nb]
    m1 = x[nb:nb+8]
    m2 = x[nb+8:nb+16]
    m3 = x[nb+16:nb+24]

    R_ts = cal_R_t(t_s, lambda_c, lambda_b, o_b, k_b)
    M_ts = cal_M_t(t_s, mu_c, mu_b, o_b)
    Phi_k = 1 - cal_phi_or_T_k(beta[c_idx], lambda_c, beta[b_idx], lambda_b, o_b, k_b)
    D_t = cal_D_t(R_ts, Phi_k, M_ts)

    MTBF = cal_MTBF_S(lambda_c, lambda_b, o_b, k_b)
    MTTR = cal_MTTR_S(mu_c, T_K_c, lambda_c, mu_b, T_K_b, lambda_b, o_b, k_b)
    ft = cal_f_cb(t_d, lambda_cb)
    ht = cal_ht(ft)
    MMDT = cal_MMDT_S(t_d, ft[c_idx], mu_c, ft[b_idx], mu_b, o_b)
    MLDT = cal_MLDT_S(ht, t_0, lambda_cb, np.concatenate([o_cb_bound[c_idx], o_b]),
                      m1, m2, m3, n_p, t_p1, t_p2)
    MPMDT = cal_MPMDT_S()
    P_ORS = t_d / (t_d + MMDT + MLDT + MPMDT)

    # 系统总质量
    total_M = np.dot(o_b, mass[b_idx]) + np.sum(mass[c_idx])
    # ret = [P_ORS * D_t, P_ORS, MTTR, MTBF, MMDT, MLDT, Phi_k, total_M]
    return -P_ORS * D_t


def con_por(x):
    nb = 3
    o_b = x[0:nb]
    m1 = x[nb:nb+8]
    m2 = x[nb+8:nb+16]
    m3 = x[nb+16:nb+24]

    ft = cal_f_cb(t_d, lambda_cb)
    ht = cal_ht(ft)
    MMDT = cal_MMDT_S(t_d, ft[c_idx], mu_c, ft[b_idx], mu_b, o_b)
    MLDT = cal_MLDT_S(ht, t_0, lambda_cb, np.concatenate([o_cb_bound[c_idx], o_b]),
                      m1, m2, m3, n_p, t_p1, t_p2)
    MPMDT = cal_MPMDT_S()
    P_ORS = t_d / (t_d + MMDT + MLDT + MPMDT)
    return 0.85 - P_ORS

def con_mttr(x):
    nb = 3
    o_b = x[0:nb]
    MTTR = cal_MTTR_S(mu_c, T_K_c, lambda_c, mu_b, T_K_b, lambda_b, o_b, k_b)
    return MTTR - 0.5

def con_mtbf(x):
    nb = 3
    o_b = x[0:nb]
    MTBF = cal_MTBF_S(lambda_c, lambda_b, o_b, k_b)
    return 600 - MTBF

def con_mmdt(x):
    nb = 3
    o_b = x[0:nb]
    ft = cal_f_cb(t_d, lambda_cb)
    MMDT = cal_MMDT_S(t_d, ft[c_idx], mu_c, ft[b_idx], mu_b, o_b)
    return MMDT - 5

def con_mldt(x):
    nb = 3
    o_b = x[0:nb]
    m1 = x[nb:nb+8]
    m2 = x[nb+8:nb+16]
    m3 = x[nb+16:nb+24]
    ft = cal_f_cb(t_d, lambda_cb)
    ht = cal_ht(ft)
    MLDT = cal_MLDT_S(ht, t_0, lambda_cb, np.concatenate([o_cb_bound[c_idx], o_b]),
                      m1, m2, m3, n_p, t_p1, t_p2)
    return MLDT - 100

def con_phi(x):
    nb = 3
    o_b = x[0:nb]
    Phi_k = 1 - cal_phi_or_T_k(beta[c_idx], lambda_c, beta[b_idx], lambda_b, o_b, k_b)
    return 0.85 - Phi_k

def con_mass(x):
    nb = 3
    o_b = x[0:nb]
    total_M = np.dot(o_b, mass[b_idx]) + np.sum(mass[c_idx])
    return total_M - 100

less_cons = [con_mass, con_mldt, con_mmdt, con_mtbf, con_mttr, con_phi, con_por]
lb = np.concatenate([np.ones(3), np.zeros(24)])
ub = np.concatenate([o_b_bound, m1_bound, m2_bound, m3_bound])

def _generator(init_val, bounds):
    _valid = True
    _len = len(init_val)
    for i in range(_len):
        if init_val[i] > bounds[i]:
            _valid = False
            return

    def _increase(cur_val):
        for i in range(_len):
            if cur_val[i] + 1 <= bounds[i]:
                cur_val[i] = cur_val[i] + 1
                return cur_val
            else:
                cur_val[i] = 0
        _valid = False
    cur_val = int_val
    while _valid:
        yield cur_val
        cur_val = _increase(cur_val)


if __name__ == '__main__':
    upper_bounds = np.concatenate([o_b_bound, m1_bound, m2_bound, m3_bound],)
    int_val = np.concatenate([[1, 1, 1], np.zeros(24, dtype=np.int32)])
    points = good_point_init(10000, int_val, upper_bounds)
    points = np.floor(points)
    points = points.astype(np.int32)
    data = []
    i = 0
    for p in points:
        f_val = obj_fun(p)
        i += 1
        print(i)
        data.append(f_val)
    import pandas as pd
    df_frame = pd.DataFrame(data)
    excel_writer = pd.ExcelWriter(r'./output/good_val.xlsx', engine="xlsxwriter")
    df_frame.to_excel(excel_writer)
    excel_writer.save()



