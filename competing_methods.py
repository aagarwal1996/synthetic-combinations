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


from scipy.linalg import hadamard
from scipy.sparse import random
from scipy import stats
from numpy.linalg import matrix_rank
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from fancyimpute import *


def horizontal_only_fit(observed_matrix,horizontal_estimator = LassoCV(cv = 3)): 
    """
    Do a separate Lasso every unit. 
    
    Parameters
    ----------
    observed_matrix: numpy array with missing entries denoted by nan
    """
    
    imputation_matrix = deepcopy(observed_matrix)
    fourier_characteristic_matrix = hadamard(observed_matrix.shape[1]) #2^number of items
    donor_unit_coefficients = []        
    for i in tqdm(range(observed_matrix.shape[0])):
        unit_outcomes = imputation_matrix[i,:]
        non_nan_indices = np.argwhere(~np.isnan(unit_outcomes))
        non_nan_indices = [non_nan_indices[index][0] for index in range(len(non_nan_indices))]
                
        unit_fourier_characteristic_matrix = fourier_characteristic_matrix[non_nan_indices,:] #X matrix used for donor unit
        unit_observed_outcomes = unit_outcomes[non_nan_indices]
        unit_lasso_reg = horizontal_estimator.fit(unit_fourier_characteristic_matrix,unit_observed_outcomes)
               
        imputation_matrix[i,:] = unit_lasso_reg.predict(fourier_characteristic_matrix)
    print(np.isnan(imputation_matrix).any())
    return imputation_matrix
    
def soft_impute_matrix_completion(observed_matrix,rank):
    """
    Complete matrix completion via soft imputation  . 
    """
    return SoftImpute(max_rank = rank,verbose = False).fit_transform(deepcopy(observed_matrix))

def MatrixFactorization_matrix_completion(observed_matrix,rank):
    """
    Complete matrix completion via matrix factorization. 
    """
    return MatrixFactorization(rank = rank,verbose = False).fit_transform(deepcopy(observed_matrix))

def IterativeSVD_matrix_completion(observed_matrix,rank):
    """
    Complete matrix completion via Iterative SVD. 
    """
    return IterativeSVD(rank = rank,verbose = False).fit_transform(deepcopy(observed_matrix))
    

    