# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:39 2016

@author: fskPioDwo
"""

#%%
from pprint import pprint

#from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from model_evaluation import MyOwnGridSearchCV, multiscorer
from model_evaluation import get_best_models_fit_results_based_on_training_data, print_best_model_summary_based_on_training_data

from os import path

#%%

#||
action_name = "Run Random Forest"
action_description = ""
action_input = {"trainX", "trainY", "fold_indexes", "rf_param_grid"}
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
    "n_jobs": 16,
    "verbose": 0
}

#param_grid= {
#     "n_estimators": [200, 400],
#     "max_depth": [4, 6]
#}

#Reduced


#Full
#param_grid= {
#     "n_estimators": [200, 400, 600, 800, 1000, 1200],
#     "max_depth": [4, 6, 8, 10, 12],
#     "max_features": ["sqrt", "log2"],
#     "criterion": ["gini", "entropy"]
#}
tmp_file_path = path.join(tmp_dir_path, pipeline_action_prefix + "__fit_results_tmp.pickle")

grid_model_param_search = MyOwnGridSearchCV(RandomForestClassifier, num_k_folds, rf_param_grid, shared_params, multiscorer, fold_indexes=fold_indexes)
fit_results = grid_model_param_search.parallel_fit(trainX, trainY, tmp_dir_path=tmp_dir_path, pipeline_action_prefix=pipeline_action_prefix, num_processes=3)

best_models_fit_results = get_best_models_fit_results_based_on_training_data(fit_results)
print_best_model_summary_based_on_training_data(best_models_fit_results)
