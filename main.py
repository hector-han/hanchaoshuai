from func import obj_fun, less_cons, lb, ub
from fpa import FlowerPollinationAlgorithm

if __name__ == '__main__':
    fpa = FlowerPollinationAlgorithm(obj_fun, 27, less_cons=less_cons, lb=lb, ub=ub,
                                     num_popu=80, N_iter=100, integer_op=True, coef=1.0, name='hanchaoshuai')
    fpa.train()
    print(fpa.x_best_list)
    fpa.save('./output')
