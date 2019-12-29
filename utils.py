import numpy as np


def _is_prime(num: int) -> bool:
    """
    判断num是否为素数
    :param num:
    :return:
    """
    assert num > 0
    # If given number is greater than 1
    if num == 1:
        return True

    # Iterate from 2 to n / 2
    for i in range(2, num // 2):
        # If num is divisible by any number between
        # 2 and n / 2, it is not prime
        if (num % i) == 0:
            return False
    else:
        return True


def _least_prime_ge(num: int) -> int:
    """
    找到大于等于num的最小素数
    :param num:
    :return:
    """
    if num <= 1:
        return 1

    i = num
    while not _is_prime(i):
         i += 1
    return i


def good_point_init(num_of_points: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    使用佳点集，在[lb, yb]中产生初始点
    :param num_of_points: 要产生的点的个数
    :param lb: 区域的下边界，如[0,0,0]
    :param ub: 区域的上边界, 如[1,1,1]
    :return: matrix, 每行是一个点，共num_of_points行
    """
    assert len(lb.shape) == 1 and lb.shape == ub.shape
    _dims = lb.shape[0]
    for i in range(_dims):
        assert lb[i] < ub[i]

    scale = ub - lb

    t = _least_prime_ge(2 * _dims + 3)
    gamma = np.arange(1, _dims + 1)
    gamma = (2 * np.pi / t) * gamma
    gamma = 2 * np.cos(gamma)
    init_points = np.zeros([num_of_points, _dims])
    for i in range(num_of_points):
        tmp = i * gamma
        init_points[i] = lb + (tmp - np.floor(tmp)) * scale
    return init_points


def random_point_init(num_of_points: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    使用佳点集，在[lb, yb]中产生初始点
    :param num_of_points: 要产生的点的个数
    :param lb: 区域的下边界，如[0,0,0]
    :param ub: 区域的上边界, 如[1,1,1]
    :return: matrix, 每行是一个点，共num_of_points行
    """
    assert len(lb.shape) == 1 and lb.shape == ub.shape
    _dims = lb.shape[0]
    for i in range(_dims):
        assert lb[i] < ub[i]
    scale = ub - lb
    init_points = np.random.random([num_of_points, _dims])
    for i in range(num_of_points):
        init_points[i] = lb + init_points[i] * scale
    return init_points


def deb_feasible_compare(xs, obj_fun, equ_cons, less_cons):
    """
    对于极小化问题，使用deb 可行性比较发判断解的优先级
    :param xs: 一系列待确定的解
    :param obj_fun: 目标函数
    :param equ_cons: 一系列 等式约束
    :param less_cons: 一系列 小于0 的不等式约束
    :return: xs中最优解的下标
    """
    delta = 1e-4
    obj_vals = [obj_fun(x) for x in xs]
    max_val = max(obj_vals)
    # 约束违反度
    phis = []
    _comparator = []
    _flag = False
    for i, x in enumerate(xs):
        eq_phi = sum([max(0, abs(equ_con(x)) - delta) for equ_con in equ_cons])
        less_phi = sum([max(0, less_con(x)) for less_con in less_cons])
        phi = eq_phi + less_phi
        phis.append(phi)
        if phi < delta:
            # 可行解
            _flag = True
            _comparator.append(obj_vals[i])
        else:
            # 不可行解一定差于可行解，固定不可行解目标函数值为max_val，这样比较他们的约束违反度即可
            _comparator.append(max_val + phi)

    indices = list(range(len(xs)))
    sorted_indices = sorted(indices, key=lambda i: _comparator[i])
    return sorted_indices[0], _flag






