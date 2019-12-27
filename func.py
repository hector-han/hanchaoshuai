
def E(P, D):
    """
    系统效能函数
    :param P: 可用度函数
    :param D: 系统可信度函数
    :return:
    """
    return P * D


def D_t(R_t, phi_k, M_t):
    """
    任务可信度函数
    :param R_t:装备可靠度函数
    :param phi_k:故障检测个利率
    :param M_t: 发生故障后的维修度函数
    :return:
    """
    tmp = R_t + (1 - R_t) * phi_k * M_t
    return tmp


def


