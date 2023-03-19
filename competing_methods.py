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
    
    observed_matrix = deepcopy(observed_matrix)
    fourier_characteristic_matrix = hadamard(observed_matrix.shape[1]) #2^number of items
    donor_unit_coefficients = []        
    for i in tqdm(range(observed_matrix.shape[0])):
        unit_outcomes = observed_matrix[i,:]
        non_nan_indices = np.argwhere(~np.isnan(unit_outcomes))
        non_nan_indices = [non_nan_indices[index][0] for index in range(len(non_nan_indices))]
                
        unit_fourier_characteristic_matrix = fourier_characteristic_matrix[non_nan_indices,:] #X matrix used for donor unit
        unit_observed_outcomes = unit_outcomes[non_nan_indices]
        unit_lasso_reg = horizontal_estimator.fit(unit_fourier_characteristic_matrix,unit_observed_outcomes)
               
        observed_matrix[i,:] = unit_lasso_reg.predict(fourier_characteristic_matrix)
    return observed_matrix
    
def soft_impute_matrix_completion(observed_matrix,rank):
    return SoftImpute(max_rank = rank).fit_transform(deeopcopy(observed_matrix))

def MatrixFactorization_matrix_completion(observed_matrix,rank):
    return MatrixFactorization(rank = r).fit_transform(deepcopy(observed_matrix))

def IterativeSVD(observed_matrix,rank):
    return IterativeSVD(rank = r).fit_transform(deepcopy(observed_matrix))
    

    