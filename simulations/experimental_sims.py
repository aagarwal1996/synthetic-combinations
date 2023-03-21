import numpy as np
import sys
from copy import deepcopy
from typing import List
import sklearn
import random
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.linalg import hadamard
from sklearn import preprocessing
from scipy.sparse import random
from random import choices

sys.path.append("../")
from competing_methods import *
from synthetic_combinations import *
import pickle           
import os

N = 100
r  = 3
#p = [7,8,9,10] #11,12,13,14,15
p = [11,12,13,14]
s = [int(i**1.5) for i in p]
mean_error_dict = {}
std_error_dict = {}

mean_donor_unit_error_dict = {}
std_donor_unit_error_dict = {}


mean_non_donor_unit_error_dict = {}
std_non_donor_unit_error_dict = {}


num_donor_units = 2*r


noise_level = 0.1

heritability = 0.6


num_reps = 5
methods = ['lasso','synth_combo','soft_impute','matrix_factorization','iterativeSVD']


def generate_alpha_matrix(N,r,p,s):
    rvs = stats.norm().rvs
    rank_users = []
    for i in range(r):
        alpha_u = random(2**p,1,density=(s)/(2**p),data_rvs = rvs).toarray()
        rank_users.append(alpha_u[:,0])
    rank_users = np.array(rank_users)
    B = np.random.dirichlet([1]*r, N - r)
    remaining_users = np.matmul(B,rank_users)
    all_users = np.concatenate((rank_users,remaining_users))
    all_users = normalize(all_users, axis=1, norm='l2')
    fourier_characteristic_matrix = hadamard(2**p)
    potential_outcome_matrix = np.matmul(all_users,fourier_characteristic_matrix)
    return all_users, potential_outcome_matrix

def get_mean_std(methods,results_dict):
    results_array = np.array(results_dict)
    results_mean = np.mean(results_array,axis = 0)
    results_mean = dict(zip(methods, results_mean))
    results_stds = np.std(results_dict,axis = 0)
    results_stds = dict(zip(methods, results_stds))
    return results_mean,results_stds
   

all_errors = []
for (i,num_interventions) in enumerate(p):
    print("running for: " + str(num_interventions)) 
    total_error_dict = []
    donor_unit_error_dict = []
    non_donor_unit_error_dict = []
    for rep in range(num_reps):
        np.random.seed(rep)
        alpha_matrix,potential_outcome_matrix = generate_alpha_matrix(N,r,num_interventions,s[i])
        if heritability is not None:
            noise_level = ((np.var(potential_outcome_matrix)*(1.0 - heritability))/heritability)**0.5
            print(noise_level)
        else:
            noise_level = noise_level
        noisy_potential_outcome_matrix = potential_outcome_matrix + np.random.normal(loc = 0,scale = noise_level,size = (N,2**num_interventions))

        num_donor_unit_observations = s[i]*num_interventions
        num_non_donor_unit_observations = 2*r**4
        observation_matrix = np.empty((N,2**num_interventions,))
        observation_matrix[:] = np.nan
        donor_unit_observations = choices(range(2**num_interventions), k = num_donor_unit_observations)
        non_donor_unit_observations = choices(range(2**num_interventions), k = num_non_donor_unit_observations)
        observation_matrix[:num_donor_units,donor_unit_observations] = noisy_potential_outcome_matrix[:num_donor_units,donor_unit_observations]
        observation_matrix[num_donor_units:,non_donor_unit_observations] = noisy_potential_outcome_matrix[num_donor_units:,non_donor_unit_observations]
        all_method_r2_score = []
        donor_unit_method_r2_score = []
        non_donor_unit_r2_score = []
        for m in methods:
            imputed_matrix = np.empty((potential_outcome_matrix.shape[0],potential_outcome_matrix.shape[1]))
            if m == 'lasso':
                lasso_matrix = horizontal_only_fit(deepcopy(observation_matrix))
                all_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix),deepcopy(lasso_matrix)))
                donor_unit_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[:num_donor_units,:]),deepcopy(lasso_matrix[:num_donor_units,:])))
                non_donor_unit_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[num_donor_units:,:]),deepcopy(lasso_matrix[num_donor_units:,:])))
                #print(np.isnan(imputed_matrix).any())
            elif m == 'synth_combo':
                sc = synth_combo()
                donor_unit_indices = [i for i in range(num_donor_units)]
                horizontal_regression_observation_matrix = sc.horizontal_fit(deepcopy(observation_matrix),donor_unit_indices)
                vertical_regression_observation_matrix = sc.vertical_fit(deepcopy(horizontal_regression_observation_matrix),donor_unit_indices,use_cv=True)
                
                all_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix),deepcopy(vertical_regression_observation_matrix)))
                
                donor_unit_potential_outcomes = deepcopy(potential_outcome_matrix[:num_donor_units,:])
                donor_unit_estimated_outcomes =  deepcopy(vertical_regression_observation_matrix[:num_donor_units,:])
                
                non_donor_unit_potential_outcomes = deepcopy(potential_outcome_matrix[num_donor_units:,:])
                non_donor_unit_estimated_outcomes = deepcopy(vertical_regression_observation_matrix[num_donor_units:,:])
                
                donor_unit_method_r2_score.append(mean_squared_error(donor_unit_potential_outcomes,donor_unit_estimated_outcomes))
                non_donor_unit_r2_score.append(mean_squared_error(non_donor_unit_potential_outcomes,non_donor_unit_estimated_outcomes))
                
            elif m == 'soft_impute':
                soft_impute_matrix = soft_impute_matrix_completion(deepcopy(observation_matrix),r)
                all_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix),deepcopy(soft_impute_matrix)))
                donor_unit_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[:num_donor_units,:]), deepcopy(soft_impute_matrix[:num_donor_units,:])))
                non_donor_unit_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[num_donor_units:,:]),deepcopy(soft_impute_matrix[num_donor_units:,:])))
                
            elif m == 'matrix_factorization':
                matrix_factorization_matrix = MatrixFactorization_matrix_completion(deepcopy(observation_matrix),r)
                all_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix),deepcopy(matrix_factorization_matrix)))
                donor_unit_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[:num_donor_units,:]), deepcopy(matrix_factorization_matrix[:num_donor_units,:])))
                non_donor_unit_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[num_donor_units:,:]),deepcopy(matrix_factorization_matrix[num_donor_units:,:])))
                
            elif m == 'iterativeSVD':
                SVD_matrix = IterativeSVD_matrix_completion(deepcopy(observation_matrix),r)
                all_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix),deepcopy(SVD_matrix)))
                donor_unit_method_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[:num_donor_units,:]), deepcopy(SVD_matrix[:num_donor_units,:])))
                non_donor_unit_r2_score.append(mean_squared_error(deepcopy(potential_outcome_matrix[num_donor_units:,:]),deepcopy(SVD_matrix[num_donor_units:,:])))
                
                
                
        total_error_dict.append(all_method_r2_score)
        donor_unit_error_dict.append(donor_unit_method_r2_score)
        non_donor_unit_error_dict.append(non_donor_unit_r2_score)
        
    #error_dict = np.array(error_dict)
    #error_means = np.mean(error_dict,axis = 0)
    #error_means = dict(zip(methods, error_means))
    #error_stds = np.std(error_dict,axis = 0)
    #error_stds = dict(zip(methods, error_stds))
    error_means, error_stds = get_mean_std(methods,total_error_dict)
    donor_unit_error_means, donor_unit_error_stds = get_mean_std(methods,donor_unit_error_dict)
    non_donor_unit_error_means, non_donor_unit_error_stds = get_mean_std(methods,non_donor_unit_error_dict)
    

    mean_error_dict[num_interventions] = error_means
    std_error_dict[num_interventions] = error_stds
    
    mean_donor_unit_error_dict[num_interventions] = donor_unit_error_means
    std_donor_unit_error_dict[num_interventions] = donor_unit_error_stds
    
    mean_non_donor_unit_error_dict[num_interventions] = non_donor_unit_error_means
    std_non_donor_unit_error_dict[num_interventions] = non_donor_unit_error_stds
    
    
    
    num_intervention_results = [error_means,error_stds]
    num_intervention_donor_results = [donor_unit_error_means,donor_unit_error_stds]
    num_intervention_non_donor_results = [non_donor_unit_error_means,non_donor_unit_error_stds]
    
    num_interventions_all_results = [num_intervention_results,num_intervention_donor_results,num_intervention_non_donor_results]
    
    os.makedirs('results/experimental_results/'+str(heritability)+'_heritability/', exist_ok = True) 
    
    with open('results/experimental_results/'+str(heritability)+'_heritability/'+str(num_interventions)+'_results.pickle','wb') as handle:
        pickle.dump(num_interventions_all_results,handle,protocol = pickle.HIGHEST_PROTOCOL)
    
    
    print("finished running for: " + str(num_interventions)) 
    
all_results = [mean_error_dict,std_error_dict]
    
with open('results/all_experimental_results.pickle', 'wb') as handle:
    pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
