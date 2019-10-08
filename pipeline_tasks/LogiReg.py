# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:39 2016

@author: fskPioDwo
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:39 2016

@author: fskPioDwo
"""

#%%
from pprint import pprint
from os import path

from sklearn.linear_model import LogisticRegression

from model_evaluation import MyOwnGridSearchCV, multiscorer
from model_evaluation import get_best_models_fit_results_based_on_training_data, print_best_model_summary_based_on_training_data

#%%

#||
action_name = "Run Logistic Regression"
action_description = ""
action_input = {"trainX", "trainY", "fold_indexes", "logireg_param_grid"}
action_output = {"fit_results"}
#||


#%%

#trainX = np.load(dp + "count_PCA__trainX.npy")
#testX = np.load(dp + "count_PCA__testX.npy")
#trainY = np.load(dp + "trainY.npy")
#testY = np.load(dp + "testY.npy")
#trainX = trainXcounts_PCA #|| input
#testX = testXcounts_PCA #|| input

num_k_folds = 3 #|| input

shared_params = {
    "random_state": 0,
    "n_jobs": 1,
    "verbose": 0,
    "solver": "lbfgs",
    "fit_intercept": True,
    "multi_class": "ovr",
    "penalty": "l2",
}



grid_model_param_search = MyOwnGridSearchCV(LogisticRegression, num_k_folds, logireg_param_grid, shared_params, multiscorer, fold_indexes=fold_indexes)
fit_results = grid_model_param_search.single_process_fit(trainX, trainY, tmp_dir_path=tmp_dir_path, pipeline_action_prefix=pipeline_action_prefix)
# Parallel fit doesn't work with current sklearn

best_models_fit_results = get_best_models_fit_results_based_on_training_data(fit_results)
print_best_model_summary_based_on_training_data(best_models_fit_results)
