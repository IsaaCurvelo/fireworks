import sys
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
        all_sparks = np.array(fireworks)
        
        for i in range(n):
            # compute si
            si = m * (np.max(fitness) - fitness[i] + epsilon) / (np.sum(max(fitness) - fitness) + epsilon)
            print('\nindividual: ' + str(fireworks[i, :]))
            print('fitness: ' + str(fitness[i]))            
            
            # bound si to am and bm
            if si < a * m:
                si = int(round(a * m))
            elif si > b * m:
                si = int(round(b * m))
            else:
                si = int(round(si))
            
            print('number of sparks: ' + str(si))
            
            # compute A
            ai = big_a_hat * (fitness[i] - np.min(fitness) + epsilon) / (np.sum(fitness - np.min(fitness)) + epsilon)
            
            print('explosion radius: ' + str(ai))
            sparks_i = np.zeros((si, d))

            for s in range(si):
                sparks_i[s, :] = fireworks[i, :]
                
                z = round(d * np.random.random())
                z = np.random.choice(range(d), z, replace=False)
                
                print('\ndimensions to be affected(' + str(len(z)) +'): ' + str(z))
                
                # perform linear displacement
                h = ai * np.random.uniform(-1, 1)
                sparks_i[s, z] = sparks_i[s, z] + h
                
                # map sparks back to the search space
                idx = np.where(sparks_i[s, :] < lwr_bnd)
                sparks_i[s, idx] = lwr_bnd[idx]
                idx = np.where(sparks_i[s, :] > upp_bnd)
                sparks_i[s, idx] = upp_bnd[idx]
                
                print('spark [' + str(s) +']: ' + str(sparks_i[s, :]))
            
#            print('sparks generated: ' + str(sparks_i))
                
#            np.concatenate((s, n), axis = 0)
            all_sparks = np.concatenate((all_sparks, sparks_i), axis = 0)
        
        idx = np.random.choice(range(len(all_sparks)), mg, replace = True)
        print("--->" + str(idx))
        print(len(all_sparks))
        print(all_sparks[idx, :])
            
            
            
    return fireworks, fitness



dimensions = 5
np.random.seed(1)

lwr_bnd = np.repeat(-5, dimensions)
upp_bnd = np.repeat(5, dimensions)

fa(rosen, lwr_bnd, upp_bnd, iterations=1, d=dimensions)
