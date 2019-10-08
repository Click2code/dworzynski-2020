# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:00:28 2019

@author: fskPioDwo
"""

#%%
from pprint import pprint

#from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

from model_evaluation import MyOwnGridSearchCV, multiscorer, get_train_test_val_stats_for_model
from model_evaluation import get_best_models_fit_results_based_on_training_data, print_best_model_summary_based_on_training_data
from model_evaluation import plot_prformance_curves_for_metric, use_only_top_X_predictions, print_test_stats_for_model
from model_evaluation import calculate_ppv_at_X, asssemble_score_dataframe


from common import load_input, start_stop_timer, my_describe, ready_for_json, dict_to_str

def save_to_value_store(value_name, value):
    try:
        value_store[value_name] = value
    except Exception :
        pass

def publish_value(value_name, value):
    if type(value) == dict:
        json_ready_dict = ready_for_json(value)

        pprint(f"{value_name}: {dict_to_str(json_ready_dict)}")
        save_to_value_store(value_name, json_ready_dict)
    else:
        pprint(f"{value_name}: {va/lue}")
        save_to_value_store(value_name, value)

p = publish_value

load_input_run_name = "T2D_timesplit__CKD_N17-N19__v14"
load_input_shared_storage_name = "T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA"

#%%
#||
action_name = "Calculate method AUROC diffs"
action_description = ""
action_input = {}
action_output = {"method_auroc_diff_CIs"}
#||

#%%

gb_fit_results = load_input(load_input_run_name, "ExGradientBoost_count__fit_results", load_input_shared_storage_name) #|| input
lr_fit_results = load_input(load_input_run_name, "LR_count__fit_results", load_input_shared_storage_name) #|| input
rf_fit_results = load_input(load_input_run_name, "RandFor_count__fit_results", load_input_shared_storage_name) #|| input
bs_fit_results = load_input(load_input_run_name, "LR_aux__fit_results", load_input_shared_storage_name) #|| input

#gb_model_summary = load_input(load_input_run_name, "ExGradientBoost_count__model_type_summary", load_input_shared_storage_name) #|| input
#lr_model_summary = load_input(load_input_run_name, "LR_count__model_type_summary", load_input_shared_storage_name) #|| input
#rf_model_summary = load_input(load_input_run_name, "RandFor_count__model_type_summary", load_input_shared_storage_name) #|| input
#bs_model_summary = load_input(load_input_run_name, "LR_aux__model_type_summary", load_input_shared_storage_name) #|| input

aux_testX = load_input(load_input_run_name, "aux__testX", load_input_shared_storage_name) #|| input
count_testX = load_input(load_input_run_name, "count__testX", load_input_shared_storage_name) #|| input
testY = load_input(load_input_run_name, "testY", load_input_shared_storage_name) #|| input

aux_valX = load_input(load_input_run_name, "aux__valX", load_input_shared_storage_name) #|| input
count_valX = load_input(load_input_run_name, "count__valX", load_input_shared_storage_name) #|| input
valY = load_input(load_input_run_name, "valY", load_input_shared_storage_name) #|| input

trainX_index__to__patient_feature_vector_indexes =  load_input(load_input_run_name, "trainX_index__to__patient_feature_vector_indexes", load_input_shared_storage_name) #|| input
testX_index__to__patient_feature_vector_indexes =  load_input(load_input_run_name, "testX_index__to__patient_feature_vector_indexes", load_input_shared_storage_name) #|| input
valX_index__to__patient_feature_vector_indexes =  load_input(load_input_run_name, "valX_index__to__patient_feature_vector_indexes", load_input_shared_storage_name) #|| input
patient_feature_vector___index__to__lbnr__list =  load_input(load_input_run_name, "patient_feature_vector___index__to__lbnr__list", load_input_shared_storage_name) #|| input
patient_feature_vector___index__to__feature__list =  load_input(load_input_run_name, "patient_feature_vector___index__to__feature__list", load_input_shared_storage_name) #|| input
subpop_basic_df = load_input(load_input_run_name, "subpop_basic_df", load_input_shared_storage_name) #|| input
best_model_criterion = "ROC AUC" #|| input


#%%

bs_best_model =  get_best_models_fit_results_based_on_training_data(bs_fit_results)[best_model_criterion]['combined_model']
lr_best_model =  get_best_models_fit_results_based_on_training_data(lr_fit_results)[best_model_criterion]['combined_model']
rf_best_model =  get_best_models_fit_results_based_on_training_data(rf_fit_results)[best_model_criterion]['combined_model']
gb_best_model =  get_best_models_fit_results_based_on_training_data(gb_fit_results)[best_model_criterion]['combined_model']

bs_calib_best_model = CalibratedClassifierCV(bs_best_model, method="sigmoid", cv="prefit").fit(aux_testX, testY)
lr_calib_best_model = CalibratedClassifierCV(lr_best_model, method="sigmoid", cv="prefit").fit(count_testX, testY)
rf_calib_best_model = CalibratedClassifierCV(rf_best_model, method="sigmoid", cv="prefit").fit(count_testX, testY)
gb_calib_best_model = CalibratedClassifierCV(gb_best_model, method="sigmoid", cv="prefit").fit(count_testX, testY)

#%%

bs_val_pred_proba = bs_best_model.predict_proba(aux_valX)[:,1]
lr_val_pred_proba = lr_best_model.predict_proba(count_valX)[:,1]
rf_val_pred_proba = rf_best_model.predict_proba(count_valX)[:,1]
gb_val_pred_proba = gb_best_model.predict_proba(count_valX)[:,1]

#Assemble score dataframe
val_score_df = pd.DataFrame(np.hstack((bs_val_pred_proba.reshape((-1,1)),
                                   lr_val_pred_proba.reshape((-1,1)),
                                   rf_val_pred_proba.reshape((-1,1)),
                                   gb_val_pred_proba.reshape((-1,1)),
                                   valY.reshape((-1,1)))),
                        columns=["bs", "lr", "rf", "gb", "y_true"])

#%%
n_samples = 1000

population_indexes = val_score_df.index

lr_bs_sample_auroc_diff = np.zeros(n_samples)
rf_bs_sample_auroc_diff = np.zeros(n_samples)
rf_lr_sample_auroc_diff = np.zeros(n_samples)
gb_bs_sample_auroc_diff = np.zeros(n_samples)
gb_lr_sample_auroc_diff = np.zeros(n_samples)
gb_rf_sample_auroc_diff = np.zeros(n_samples)

for sample_i in range(n_samples):
        #print(sample_i)
        sample_indexes = np.random.choice(population_indexes, size=population_indexes.shape, replace=True) #sampling with replacement
        sample_score_df = val_score_df.loc[sample_indexes]
        if np.sum(sample_score_df.loc[:, "y_true"]) < 1:
            sample_i -= 1
            continue

        bs_sample_auroc = roc_auc_score(y_true=sample_score_df.loc[:, "y_true"], y_score=sample_score_df.loc[:, "bs"])
        lr_sample_auroc = roc_auc_score(y_true=sample_score_df.loc[:, "y_true"], y_score=sample_score_df.loc[:, "lr"])
        rf_sample_auroc = roc_auc_score(y_true=sample_score_df.loc[:, "y_true"], y_score=sample_score_df.loc[:, "rf"])
        gb_sample_auroc = roc_auc_score(y_true=sample_score_df.loc[:, "y_true"], y_score=sample_score_df.loc[:, "gb"])

        lr_bs_sample_auroc_diff[sample_i] = lr_sample_auroc - bs_sample_auroc
        rf_bs_sample_auroc_diff[sample_i] = rf_sample_auroc - bs_sample_auroc
        rf_lr_sample_auroc_diff[sample_i] = rf_sample_auroc - lr_sample_auroc
        gb_bs_sample_auroc_diff[sample_i] = gb_sample_auroc - bs_sample_auroc
        gb_lr_sample_auroc_diff[sample_i] = gb_sample_auroc - lr_sample_auroc
        gb_rf_sample_auroc_diff[sample_i] = gb_sample_auroc - rf_sample_auroc

lr_bs_sample_auroc_diff = np.sort(lr_bs_sample_auroc_diff)
rf_bs_sample_auroc_diff = np.sort(rf_bs_sample_auroc_diff)
rf_lr_sample_auroc_diff = np.sort(rf_lr_sample_auroc_diff)
gb_bs_sample_auroc_diff = np.sort(gb_bs_sample_auroc_diff)
gb_lr_sample_auroc_diff = np.sort(gb_lr_sample_auroc_diff)
gb_rf_sample_auroc_diff = np.sort(gb_rf_sample_auroc_diff)

def get_confidence_intervals(diffs):
    sorted_diffs = np.sort(diffs)
    return {
            "median": sorted_diffs[500],
            "mean": np.mean(sorted_diffs),
            "upper": sorted_diffs[975],
            "lower": sorted_diffs[25]
            }

method_auroc_diff_CIs = {
        "lr_bs_auroc_diff_CI": get_confidence_intervals(lr_bs_sample_auroc_diff),
        "rf_bs_auroc_diff_CI": get_confidence_intervals(rf_bs_sample_auroc_diff),
        "rf_lr_auroc_diff_CI": get_confidence_intervals(rf_lr_sample_auroc_diff),
        "gb_bs_auroc_diff_CI": get_confidence_intervals(gb_bs_sample_auroc_diff),
        "gb_lr_auroc_diff_CI": get_confidence_intervals(gb_lr_sample_auroc_diff),
        "gb_rf_auroc_diff_CI": get_confidence_intervals(gb_rf_sample_auroc_diff)
        }

#%%


