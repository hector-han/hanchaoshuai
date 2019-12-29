from func import obj_fun, lb, ub, obj_fun_and_less_cons
from fpa import FlowerPollinationAlgorithm

if __name__ == '__main__':
    fpa = FlowerPollinationAlgorithm(obj_fun, 27, less_cons=[], obj_fun_and_less_cons=obj_fun_and_less_cons, lb=lb, ub=ub,
                                     num_popu=100, N_iter=100, integer_op=True, coef=1.0, name='hanchaoshuai')
    fpa.train()
    print(fpa.x_best_list[-1])
    print(fpa.stored)
    fpa.save('./output')
