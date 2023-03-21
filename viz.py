import math
import os.path
import pickle 
import subprocess
from math import ceil
from os.path import dirname
from os.path import join as oj
from typing import List, Dict, Any, Union, Tuple
import warnings
from copy import deepcopy

# import adjustText
import dvu
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm



dvu.set_style()
mpl.rcParams['figure.dpi'] = 250

cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'

methods = ['lasso','synth_combo','soft_impute','iterativeSVD'] #'lasso','synth_combo',
p = [8,9,10,11,12,13]
#p = [8,9,10]
heritability = 0.8

method_names = {
        'synth_combo': 'Synthetic Combinations',
        'synth_combo_donor': 'Synthetic Combinations (donor set)',
        'synth_combo_non_donor': 'Synthetic Combinations (non-donor set)',
    
        'lasso': 'Lasso',
        'lasso_donor': 'Lasso (donor set)',
        'lasso_non_donor': 'Lasso (non-donor set)',
    
    
        'soft_impute': 'Soft Impute',
        'soft_impute_donor': 'Soft Impute (donor set)',
        'soft_impute_non_donor': 'Soft Impute (non-donor set)',
    
        
        'matrix_factorization': 'Matrix Factorization',
        'matrix_factorization_donor': 'Matrix Factorization (donor set)',
        'matrix_factorization_non_donor': 'Matrix Factorization (non-donor set)',

        
        'iterativeSVD': 'Iterative SVD',
        'iterativeSVD_donor': 'Iterative SVD (donor set)',
        'iterativeSVD_non_donor': 'Iterative SVD (non-donor set)',
    }



COLORS = {
        'synth_combo': 'black',
        'synth_combo_donor': 'black',
        'synth_combo_non_donor': 'black',
    
        'lasso': cg,
        'lasso_donor': cg,
        'lasso_non_donor': cg,
    
    
        'soft_impute': cr,
        'soft_impute_donor': cr,
        'soft_impute_non_donor': cr,
    
        
        'matrix_factorization': cp,
        'matrix_factorization_donor': cp,
        'matrix_factorization_non_donor': cp,

        
        'iterativeSVD': cb2,
        'iterativeSVD_non_donor': cb2,
        'iterativeSVD_donor': cb2,
    }


method_line_styles = {
        'synth_combo': 'solid',
        'synth_combo_donor': 'dashed',
        'synth_combo_non_donor': 'dotted',
    
        'lasso': 'solid',
        'lasso_donor': 'dashed',
        'lasso_non_donor': 'dotted',
    
    
        'soft_impute': 'solid',
        'soft_impute_donor': 'dashed',
        'soft_impute_non_donor': 'dotted',
    
        
        'matrix_factorization': 'solid',
        'matrix_factorization_donor': 'dashed',
        'matrix_factorization_non_donor': 'dotted',

        
        'iterativeSVD': 'solid',
        'iterativeSVD_donor': 'dashed',
        'iterativeSVD_non_donor': 'dotted',
    }






alpha = 1.0
lw = 2

#path = "simulations/results/experimental_results/"+str(heritability)+"_heritability/"
path = "simulations/results/observation_results/"+str(heritability)+"_heritability/"
methods_to_plot = ['lasso',
                   #'lasso_donor',
                   #'lasso_non_donor',
                   'synth_combo',
                   'synth_combo_donor',
                   'synth_combo_non_donor', 
                   'soft_impute',
                   #'soft_impute_donor',
                   #'soft_impute_non_donor',
                   #'matrix_factorization',
                   #'matrix_factorization_donor',
                   #'matrix_factorization_non_donor',
                   'iterativeSVD',
                    #'iterativeSVD_donor',
                   #'iterativeSVD_non_donor']
                  ]

def concatenate_results():
    all_mean_error = []
    all_std_error = []
    
    donor_unit_mean_error = []
    donor_unit_std_error = []
    
    non_donor_unit_mean_error = []
    non_donor_unit_std_error = []
        
    for num_interventions in p:
        num_intervention_path =  path + str(num_interventions)+'_results.pickle' #'simulations/results/'+str(num_interventions)+'_experimental_results.pickle'
        with open(num_intervention_path, 'rb') as f:
            num_intervention_results = pickle.load(f)
        print(num_intervention_results)
        all_mean_error.append(num_intervention_results[0][0])
        all_std_error.append(num_intervention_results[0][1])
        
        donor_unit_mean_error.append(num_intervention_results[1][0])
        donor_unit_std_error.append(num_intervention_results[1][1])
        
        non_donor_unit_mean_error.append(num_intervention_results[2][0])
        non_donor_unit_std_error.append(num_intervention_results[2][1])
        
    methods = all_mean_error[0].keys()
    mean_error_dict = {}
    std_error_dict = {}
    
    mean_donor_unit_dict = {}
    std_donor_unit_dict = {}
    
    mean_non_donor_unit_dict = {}
    std_non_donor_unit_dict = {}
    
    for m in methods:
        method_mean_error = []
        method_std_error = []
        
        method_donor_unit_mean_error = []
        method_donor_unit_std_error = []
        
        method_non_donor_unit_mean_error = []
        method_non_donor_unit_std_error = []
        
        
        for i in range(len(all_mean_error)):
            method_mean_error.append(all_mean_error[i][m])
            method_std_error.append(all_std_error[i][m])
            
            method_donor_unit_mean_error.append(donor_unit_mean_error[i][m])
            method_donor_unit_std_error.append(donor_unit_std_error[i][m])
            
            method_non_donor_unit_mean_error.append(non_donor_unit_mean_error[i][m])
            method_non_donor_unit_std_error.append(non_donor_unit_std_error[i][m])
            
            
        mean_error_dict[m] = method_mean_error
        std_error_dict[m] = method_std_error
        
        mean_donor_unit_dict[m + '_donor'] = method_donor_unit_mean_error
        std_donor_unit_dict[m + '_donor'] = method_donor_unit_std_error
        
        mean_non_donor_unit_dict[m + '_non_donor'] = method_non_donor_unit_mean_error
        std_non_donor_unit_dict[m + '_non_donor'] = method_non_donor_unit_std_error
        
    mean_error_df = pd.DataFrame(mean_error_dict)
    std_error_df = pd.DataFrame(std_error_dict)
    
    mean_error_donor_unit_df = pd.DataFrame(mean_donor_unit_dict)
    std_error_donor_unit_df = pd.DataFrame(std_donor_unit_dict)
    
    mean_error_non_donor_unit_df = pd.DataFrame(mean_non_donor_unit_dict)
    std_error_non_donor_unit_df = pd.DataFrame(std_non_donor_unit_dict)
    
    return mean_error_df,std_error_df,mean_error_donor_unit_df,std_error_donor_unit_df,mean_error_non_donor_unit_df,std_error_non_donor_unit_df
        
    

def plot_error_curves(mean_error_df,std_error_df,mean_error_donor_unit_df,std_error_donor_unit_df,mean_error_non_donor_unit_df,std_error_non_donor_unit_df):
    plt.figure(facecolor = 'w')
    for method in methods_to_plot:
        if 'non_donor' in method:
            plt.errorbar(p,mean_error_non_donor_unit_df[method],yerr=std_error_non_donor_unit_df[method], linewidth = lw, alpha = alpha,
                         color = COLORS[method],marker = 'o',label = method_names[method],linestyle =method_line_styles[method])
        elif 'donor' in method:
            plt.errorbar(p,mean_error_donor_unit_df[method] + np.random.normal(scale = 0.001),yerr=std_error_donor_unit_df[method], linewidth = lw, alpha = alpha,
                         color = COLORS[method],marker = 'o',label = method_names[method],linestyle = method_line_styles[method])
        else:
            plt.errorbar(p,mean_error_df[method],yerr=std_error_df[method], linewidth = lw, alpha = alpha,
                         color = COLORS[method],marker = 'o',label = method_names[method],linestyle = method_line_styles[method])


    plt.ylabel("MSE",fontsize = 18)
    plt.yticks(fontsize = 14)
    plt.xlabel("# Interventions",fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.tight_layout()
    plt.yscale('log')
    #dvu.line_legend(fontsize=10, xoffset_spacing=0.1, adjust_text_labels=True)
    plt.legend(loc = 'upper left', fontsize = 12)
    #plt.savefig("plots/experimental_design_results.png")
    plt.savefig("plots/observational_design_results.png")
    
    
    

mean_error_df,std_error_df,mean_error_donor_unit_df,std_error_donor_unit_df,mean_error_non_donor_unit_df,std_error_non_donor_unit_df = concatenate_results()
plot_error_curves(mean_error_df,std_error_df,mean_error_donor_unit_df,std_error_donor_unit_df,mean_error_non_donor_unit_df,std_error_non_donor_unit_df)