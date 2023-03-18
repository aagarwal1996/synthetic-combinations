from copy import deepcopy
from typing import List

import numpy as np
from sklearn import datasets
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier,export_text
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LassoCV
from celer import Lasso, LassoCV

from scipy.linalg import hadamard
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class synth_combo():
    
    def __init__(self):
        pass
    
    def horizontal_fit(self,observation_matrix,donor_unit_indices,error_threshold = None,horizontal_estimator = LassoCV(cv = 3),refit = False): #self
        fourier_characteristic_matrix = hadamard(observation_matrix.shape[1]) #2^number of items
        donor_unit_coefficients = []
        if donor_unit_indices == None:
            assert error_threshold is not None
            #donor_unit_indices = self.get_donor_unit_indices(observation_matrix,error_threshold)
        else:
            for donor_unit in tqdm(donor_unit_indices):
                donor_unit_outcomes = observation_matrix[donor_unit,:]
                non_nan_indices = np.argwhere(~np.isnan(donor_unit_outcomes))
                non_nan_indices = [non_nan_indices[index][0] for index in range(len(non_nan_indices))]
                
                donor_unit_fourier_characteristic_matrix = fourier_characteristic_matrix[non_nan_indices,:] #X matrix used for donor unit
                donor_unit_observed_outcomes = donor_unit_outcomes[non_nan_indices]
                lasso_reg = horizontal_estimator.fit(donor_unit_fourier_characteristic_matrix,donor_unit_observed_outcomes)
               
                observation_matrix[donor_unit,:] = lasso_reg.predict(fourier_characteristic_matrix)
            return observation_matrix
       
     
    def vertical_fit(self,horizontal_imputed_observation_matrix,donor_unit_indices,num_components):
        fourier_characteristic_matrix = hadamard(horizontal_imputed_observation_matrix.shape[1]) #2^number of items
        N = horizontal_imputed_observation_matrix.shape[0]
        indices = set([i for i in range(N)])
        non_donor_unit_indices = list(indices.difference(set(donor_unit_indices)))
        for n in tqdm(non_donor_unit_indices):
            n_outcomes = horizontal_imputed_observation_matrix[n,:]
            non_nan_indices = np.argwhere(~np.isnan(n_outcomes))
            non_nan_indices = [non_nan_indices[index][0] for index in range(len(non_nan_indices))]
        
            n_fourier_characteristic_matrix = fourier_characteristic_matrix[non_nan_indices,:] #X matrix used for donor unit
            n_observed_outcomes = n_outcomes[non_nan_indices]
            donor_unit_outcomes = horizontal_imputed_observation_matrix[donor_unit_indices,:]
            donor_unit_n_outcomes = donor_unit_outcomes[:,non_nan_indices]
        
            pca = PCA(n_components = num_components)
            X_pca = pca.fit_transform(donor_unit_n_outcomes.T)
            regr = LinearRegression(fit_intercept=True)
            regr.fit(X_pca,n_observed_outcomes)
            X_donor_pca = pca.transform(horizontal_imputed_observation_matrix[donor_unit_indices,:].T)
            n_preds = regr.predict(X_donor_pca)
            horizontal_imputed_observation_matrix[n,:] = n_preds
        return horizontal_imputed_observation_matrix



class horizontal_matrix_completion():
    
    def __init__(self):
        pass
    
    def fit(self,observation_matrix,horizontal_estimator = LassoCV(cv = 3): 
        fourier_characteristic_matrix = hadamard(observation_matrix.shape[1]) #2^number of items
        donor_unit_coefficients = []
        if donor_unit_indices == None:
            assert error_threshold is not None
            #donor_unit_indices = self.get_donor_unit_indices(observation_matrix,error_threshold)
        else:
            for donor_unit in tqdm(observation_matrix.shape[0]):
                donor_unit_outcomes = observation_matrix[donor_unit,:]
                non_nan_indices = np.argwhere(~np.isnan(donor_unit_outcomes))
                non_nan_indices = [non_nan_indices[index][0] for index in range(len(non_nan_indices))]
                
                donor_unit_fourier_characteristic_matrix = fourier_characteristic_matrix[non_nan_indices,:] #X matrix used for donor unit
                donor_unit_observed_outcomes = donor_unit_outcomes[non_nan_indices]
                lasso_reg = horizontal_estimator.fit(donor_unit_fourier_characteristic_matrix,donor_unit_observed_outcomes)
               
                observation_matrix[donor_unit,:] = lasso_reg.predict(fourier_characteristic_matrix)
            return observation_matrix
    

     #if refit:
     #               lasso_reg_non_zero = np.nonzero(lasso_reg.coef_)[0]
     #               selected_subsets = donor_unit_fourier_characteristic_matrix[:,lasso_reg_non_zero]
     #               linear_regr = LinearRegression(fit_intercept=True)
     #               linear_regr.fit(selected_subsets,donor_unit_observed_outcomes)
     #               reg_coef = []
     #               for i in range(fourier_characteristic_matrix.shape[1]):
     #                   count = 0
     #                   if i in lasso_reg_non_zero:
     #                       reg_coef.append(linear_regr.coef_[count])
     #                       count = count + 1
     #                   else:
     #                       reg_coef.append(0.0)
     #               observation_matrix[donor_unit,:] = np.matmul(fourier_characteristic_matrix,np.array(reg_coef))
     #           else: