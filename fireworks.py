import sys
import random
import numpy as np
from scipy.optimize import rosen

epsilon = sys.float_info.epsilon

#np.flatnonzero(x<0.5)

def create_fireworks(fitness_function, lwr_bnd, upp_bnd, n, d):
    fireworks = np.random.random((n, d))
    fireworks = lwr_bnd + fireworks * (upp_bnd - lwr_bnd)
    fitness = np.apply_along_axis(fitness_function, 1, fireworks)
    return fireworks, fitness


def fa(fitness_function, lwr_bnd, upp_bnd, n = 5, d = 30,  iterations = 500,
       m = 50, big_a_hat = 40,  a = 0.04, b = 0.8, mg = 5 ):
    
    fireworks, fitness = create_fireworks(fitness_function, lwr_bnd, upp_bnd,
                                          n, d)
    for t in range(iterations):
        print("iteration " + str(t) + ":")
        for f in range(n):
            
            si = m * (np.max(fitness) - fitness[f] + epsilon) / (np.sum(max(fitness) - fitness) + epsilon)
            
            # print(si)
            
            
            if si < a * m:
                si = round(a * m)
            elif si > b * m:
                si = round(b * m)
            else:
                si = round(si)
            
            print('rounded si: ' + str(si))
            
    return fireworks, fitness



random.seed(1)
np.random.seed(1)

fa(rosen, 5, -5, iterations=1)
