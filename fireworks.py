import numpy as np
from scipy.optimize import rosen
from scipy.spatial import distance_matrix

epsilon = np.finfo(float).eps

#np.flatnonzero(x<0.5)

def create_fireworks(fitness_function, lwr_bnd, upp_bnd, n, d):
    fireworks = np.random.random((n, d))
    fireworks = lwr_bnd + fireworks * (upp_bnd - lwr_bnd)
    fitnesses = np.apply_along_axis(fitness_function, 1, fireworks)
    return fireworks, fitnesses


def fa(fitness_function, lwr_bnd, upp_bnd, n = 5, d = 30,  iterations = 500,
       m = 50, big_a_hat = 40,  a = 0.04, b = 0.8, mg = 5 ):
    
    fireworks, fitnesses = create_fireworks(fitness_function, lwr_bnd, upp_bnd,
                                          n, d)
    for t in range(iterations):
        #print("iteration " + str(t) + ":")
        all_sparks = np.array(fireworks)
        
        for i in range(n):
#             compute si
            si = m * (np.max(fitnesses) - fitnesses[i] + epsilon) /\
            (np.sum(max(fitnesses) - fitnesses) + epsilon)
            
            #print('\nindividual: ' + str(fireworks[i, :]))
            #print('fitnesses: ' + str(fitnesses[i]))            
            
#             bound si to am and bm
            if si < a * m:
                si = int(round(a * m))
            elif si > b * m:
                si = int(round(b * m))
            else:
                si = int(round(si))
            
            #print('number of sparks: ' + str(si))
            
#             compute A
            ai = big_a_hat * (fitnesses[i] - np.min(fitnesses) + epsilon) /\
            (np.sum(fitnesses - np.min(fitnesses)) + epsilon)
            
            #print('explosion radius: ' + str(ai))
            sparks_i = np.zeros((si, d))

#            compute sparks of i
            for s in range(si):
                sparks_i[s, :] = fireworks[i, :]
                
                z = round(d * np.random.random())
                z = np.random.choice(range(d), z, replace=False)
                
                #print('\naffected dimensions(' + str(len(z)) +'): ' + str(z))
                
#                 perform linear displacement
                h = ai * np.random.uniform(-1, 1)
                sparks_i[s, z] = sparks_i[s, z] + h
                
#                map sparks back to the search space
                idx = np.where(sparks_i[s, :] < lwr_bnd)
                sparks_i[s, idx] = lwr_bnd[idx]
                idx = np.where(sparks_i[s, :] > upp_bnd)
                sparks_i[s, idx] = upp_bnd[idx]
                
                #print('spark [' + str(s) +']: ' + str(sparks_i[s, :]))
            
                
            all_sparks = np.concatenate((all_sparks, sparks_i), axis = 0)
        
#        choose individuals for gaussian displacement
        idx = np.random.choice(range(len(all_sparks)), mg, replace = True)
        #print("size of all_sparks: " + str(len(all_sparks)))
        #print("performing gaussian explosions on: " + str(idx))
        
        gaussian_sparks = np.array(all_sparks[idx, :])
       
        #print("\nchosen sparks for gauss explosion : \n" + str(gaussian_sparks))
        
        for i in range(mg):
            z = round(d * np.random.random())
            z = np.random.choice(range(d), z, replace=False)
#            perform gaussian displacement
            g = np.random.normal(1, 1)
            gaussian_sparks[i, z] = gaussian_sparks[i, z] * g
            
#            map sparks back to the search space
            idx = np.where(gaussian_sparks[i, :] < lwr_bnd)
            gaussian_sparks[i, idx] = lwr_bnd[idx]
            idx = np.where(gaussian_sparks[i, :] > upp_bnd)
            gaussian_sparks[i, idx] = upp_bnd[idx]
        
        all_sparks = np.concatenate((all_sparks, gaussian_sparks), axis = 0)
        all_fitnesses = np.apply_along_axis(fitness_function, 1, all_sparks)
        
        #extract best individual
        x_star_idx = np.argmin(all_fitnesses)
        x_star = all_sparks[x_star_idx, :]
        x_star_fit = all_fitnesses[x_star_idx]
        
        #print("\nbest individual:" + str(x_star))
        #print("best fitness:" + str(x_star_fit))
        
        all_sparks = np.delete(all_sparks, x_star_idx, axis = 0)
        all_fitnesses = np.delete(all_fitnesses, x_star_idx)
        
#        compute selection probability
        overall_distance = np.sum(distance_matrix(all_sparks, all_sparks), 
                                  axis=0)
        
        p = np.divide(overall_distance, np.sum(overall_distance))
        
        #extract n-1 indexes with p
        p_indexes = np.random.choice(range(len(all_sparks)), n - 1, 
                                     replace = False, p=p)
        
        #print("\n" + str(overall_distance))
        
#        generate next population and fitnesses
        fireworks[0 ,:] =  x_star
        fireworks[range(1, n), :] = all_sparks[p_indexes, :]
        
        fitnesses[0] = x_star_fit
        fitnesses[range(1, n)] = all_fitnesses[p_indexes]
        
        
    #print("end")
    return fireworks, fitnesses



dimensions = 30
#np.random.seed(1)
lwr_bnd = np.repeat(-5, dimensions)
upp_bnd = np.repeat(5, dimensions)

fw, ft = fa(rosen, lwr_bnd, upp_bnd, d=dimensions, iterations=500, m = 50)
