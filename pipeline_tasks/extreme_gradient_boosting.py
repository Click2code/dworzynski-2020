# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:39 2016

@author: fskPioDwo
"""

#%%
from pprint import pprint
from os import path

from xgboost import DMatrix
from xgboost.sklearn import XGBClassifier

from model_evaluation import MyOwnGridSearchCV, multiscorer
from model_evaluation import get_best_models_fit_results_based_on_training_data, print_best_model_summary_based_on_training_data

#%%

#||
action_name = "Run Extreme Gradient Boosting"
action_description = ""
action_input = {"trainX", "trainY", "fold_indexes", "xgb_param_grid"}
action_output = {}
#||

#%%

#model = GradientBoostingClassifier(
#        learning_rate=0.1, n_estimators=100, max_depth=4, max_features=None,
#        min_samples_split=2, min_samples_leaf=1, subsample=1.0,
#        loss="deviance",  criterion="friedman_mse",
#        random_state=0, verbose=1)


num_k_folds = 3 #|| input

shared_params = {
    "silent": 0,
    "random_state": 0,
    "scale_pos_weight": (1.0 * len(trainX)) / sum(trainY),
    "booster": "gbtree",
    "n_jobs": 10,

    "eval_metric": "auc",
    "insert_xgb_eval_set": True,
    "additional_fit_params": {
        "eval_metric": "auc",
        "early_stopping_rounds": 10,
        "verbose": 1
    }
#    "objective": "binary:logistic", #something to consider
#    "eval_metric": "auc"
}


grid_model_param_search = MyOwnGridSearchCV(XGBClassifier, num_k_folds, xgb_param_grid, shared_params, multiscorer, fold_indexes=fold_indexes)

#fit_results = grid_model_param_search.parallel_fit(trainX, trainY, tmp_dir_path=tmp_dir_path, pipeline_action_prefix=pipeline_action_prefix, num_processes=4) #||output
fit_results = grid_model_param_search.single_process_fit(trainX, trainY, tmp_dir_path=tmp_dir_path, pipeline_action_prefix=pipeline_action_prefix) #||output

best_models_fit_results = get_best_models_fit_results_based_on_training_data(fit_results)
print_best_model_summary_based_on_training_data(best_models_fit_results)
