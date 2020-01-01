from func import lb, ub, obj_fun_and_less_cons, transformer
from fpa import FlowerPollinationAlgorithm

if __name__ == '__main__':
    fpa = FlowerPollinationAlgorithm(27, obj_fun_and_less_cons, transformer, lb=lb, ub=ub,
                                     num_popu=100, N_iter=100, integer_op=True, coef=1.0, name='hanchaoshuai')
    fpa.train()
    print('目标函数值和约束值', fpa.f_and_cons_list[-1])
    print('最优解', fpa.x_best_list[-1])
    fpa.save('./output', axis=[20, 24], axis_name=['m3_2', 'm3_6'],
             f_and_cons_name=['E', 'PORS', 'MTTR', 'MTBF', 'MMDT', 'MLDT', 'Phi_k', 'mass'])
