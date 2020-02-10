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
#        print("iteration " + str(t) + ":")
        for f in range(n):
            
            si = m * (np.max(fitness) - fitness[f] + epsilon) / (np.sum(max(fitness) - fitness) + epsilon)
            print('\nfitness: ' + str(fitness[f]))            
            
            if si < a * m:
                si = int(round(a * m))
            elif si > b * m:
                si = int(round(b * m))
            else:
                si = int(round(si))
            
            print('number of sparks: ' + str(si))
            
            
            ai = big_a_hat * (fitness[f] - np.min(fitness) + epsilon) / (np.sum(fitness - np.min(fitness)) + epsilon)
            print('explosion radius: ' + str(ai))
            
            for s in range(si):
                spark = fireworks[f, :]
                print(spark)
                
                z = round(d * random.random())
                print('dimensions to be affected: ' + str(z))
            
            
            
    return fireworks, fitness



random.seed(1)
np.random.seed(1)

fa(rosen, 5, -5, iterations=1, d=3)
