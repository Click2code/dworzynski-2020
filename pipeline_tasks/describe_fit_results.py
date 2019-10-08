# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:39 2016

@author: fskPioDwo
"""

#%%
from pprint import pprint

#from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import roc_curve, precision_recall_curve

from model_evaluation import MyOwnGridSearchCV, multiscorer, get_train_test_val_stats_for_model
from model_evaluation import get_best_models_fit_results_based_on_training_data, print_best_model_summary_based_on_training_data
from model_evaluation import plot_prformance_curves_for_metric, use_only_top_X_predictions, print_test_stats_for_model
from model_evaluation import calculate_ppv_at_X, asssemble_score_dataframe
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

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
        pprint(f"{value_name}: {value}")
        save_to_value_store(value_name, value)

p = publish_value

load_input_run_name = "T2D_timesplit__MI__v14"
#load_input_run_name = "T2D_timesplit__Stroke__v14"
load_input_shared_storage_name = "T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA"

#%%
#||
action_name = "Describe Fit Results"
action_description = ""
action_input = {"fit_results", "trainX", "testX", "trainY", "testY", "patient_feature_vector___index__to__feature__list",
                "testX_index__to__patient_feature_vector_indexes", "patient_feature_vector___index__to__lbnr__list", "subpop_basic_df"}
action_output = {"model_type_summary"}
#||

#%%
#fit_results = load_input(load_input_run_name, "ExGradientBoost_count__fit_results", load_input_shared_storage_name) #|| input
#fit_results = load_input(load_input_run_name, "LR_aux__fit_results", load_input_shared_storage_name) #|| input
fit_results = load_input(load_input_run_name, "ExGradientBoost_count__fit_results", load_input_shared_storage_name) #|| input
#trainX = load_input(load_input_run_name, "aux__trainX", load_input_shared_storage_name) #|| input
trainX = load_input(load_input_run_name, "count__trainX", load_input_shared_storage_name) #|| input
trainY = load_input(load_input_run_name, "trainY", load_input_shared_storage_name) #|| input
#testX = load_input(load_input_run_name, "aux__testX", load_input_shared_storage_name) #|| input
testX = load_input(load_input_run_name, "count__testX", load_input_shared_storage_name) #|| input
testY = load_input(load_input_run_name, "testY", load_input_shared_storage_name) #|| input
#valX = load_input(load_input_run_name, "aux__valX", load_input_shared_storage_name) #|| input
valX = load_input(load_input_run_name, "count__valX", load_input_shared_storage_name) #|| input
valY = load_input(load_input_run_name, "valY", load_input_shared_storage_name) #|| input

trainX_index__to__patient_feature_vector_indexes =  load_input(load_input_run_name, "trainX_index__to__patient_feature_vector_indexes", load_input_shared_storage_name) #|| input
testX_index__to__patient_feature_vector_indexes =  load_input(load_input_run_name, "testX_index__to__patient_feature_vector_indexes", load_input_shared_storage_name) #|| input
valX_index__to__patient_feature_vector_indexes =  load_input(load_input_run_name, "valX_index__to__patient_feature_vector_indexes", load_input_shared_storage_name) #|| input
patient_feature_vector___index__to__lbnr__list =  load_input(load_input_run_name, "patient_feature_vector___index__to__lbnr__list", load_input_shared_storage_name) #|| input
patient_feature_vector___index__to__feature__list =  load_input(load_input_run_name, "patient_feature_vector___index__to__feature__list", load_input_shared_storage_name) #|| input
subpop_basic_df = load_input(load_input_run_name, "subpop_basic_df", load_input_shared_storage_name) #|| input
best_model_criterion = "ROC AUC" #|| input
#%%


def get_deathY(subpop_basic_df, trainX_index__to__patient_feature_vector_indexes, testX_index__to__patient_feature_vector_indexes, patient_feature_vector___index__to__lbnr__list):
    trainX_index__to__lbnr_list = []
    for patient_feature_vector_index in trainX_index__to__patient_feature_vector_indexes:
        trainX_index__to__lbnr_list.append(patient_feature_vector___index__to__lbnr__list[patient_feature_vector_index])

    train_deathY = 1*subpop_basic_df.loc[trainX_index__to__lbnr_list, "died_during_followup"]

    testX_index__to__lbnr_list = []
    for patient_feature_vector_index in testX_index__to__patient_feature_vector_indexes:
        testX_index__to__lbnr_list.append(patient_feature_vector___index__to__lbnr__list[patient_feature_vector_index])

    test_deathY = 1*subpop_basic_df.loc[testX_index__to__lbnr_list, "died_during_followup"]

    valX_index__to__lbnr_list = []
    for patient_feature_vector_index in valX_index__to__patient_feature_vector_indexes:
        valX_index__to__lbnr_list.append(patient_feature_vector___index__to__lbnr__list[patient_feature_vector_index])

    val_deathY = 1*subpop_basic_df.loc[valX_index__to__lbnr_list, "died_during_followup"]

    return train_deathY, test_deathY, val_deathY

def print_summary_of_death_competing_risk(model, testX, testY, valY, train_deathY, test_deathY, val_deathY):
    y_pred_prob=model.predict_proba(testX)[:,1]
    y_pred_binary=model.predict(testX)

    summary = {}

    #For all predictions
    false_positives = (y_pred_binary - (y_pred_binary & testY))
    false_positives__and__dead = false_positives & test_deathY

    summary["total # individuals who died during followup"] = sum(test_deathY)
    summary["total # individuals"] = test_deathY.shape[0]
    summary["DFP/(P+N)"] = 1.0 * sum(test_deathY) / test_deathY.shape[0]

    summary["# false positives (FP)"] = sum(false_positives)
    summary["# false positives who died during followup (DFP)"] = sum(false_positives__and__dead)
    summary["DFP/FP"] = 1.0*sum(false_positives__and__dead) / sum(false_positives)

    true_positives = y_pred_binary & testY
    true_positives__and__dead = true_positives & test_deathY

    summary["# true positives (TP)"] = sum(true_positives)
    summary["# true positives who died during followup (DTP)"] = sum(true_positives__and__dead)
    summary["DTP/TP"] = 1.0*sum(true_positives__and__dead) / sum(true_positives)


    top_1000__y_pred_binary = use_only_top_X_predictions(y_pred_binary, y_pred_prob, 1000)

    top_1000__false_positives = (top_1000__y_pred_binary - (top_1000__y_pred_binary & testY))
    top_1000__false_positives__and__dead = top_1000__false_positives & test_deathY

    summary["# false positives (FP) in top 1000"] = sum(top_1000__false_positives)
    summary["# false positives who died during followup (DFP) in top 1000"] = sum(top_1000__false_positives__and__dead)
    summary["top 1000 DFP/FP"] = 1.0*sum(top_1000__false_positives__and__dead) / sum(top_1000__false_positives)

    top_1000__true_positives = top_1000__y_pred_binary & testY
    top_1000__true_positives__and__dead = top_1000__true_positives & test_deathY

    summary["# true positives (TP) in top 1000"] = sum(top_1000__true_positives)
    summary["# true positives who died during followup (DTP) in top 1000"] = sum(top_1000__true_positives__and__dead)
    summary["top 1000 DTP/TP"] = 1.0*sum(top_1000__true_positives__and__dead) / sum(top_1000__true_positives)

    return summary

def print_big(s):
    print(("\n\n" + "#"*len(s)))
    print(s)
    print(("#"*len(s)))

def print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, measure_name, train_deathY, test_deathY, val_deathY, calibrate_model=False):


    model_summary = {}
    print_big(f"Test results for best model according to the {measure_name} measure")
    best_model = best_models_fit_results[measure_name]['combined_model']
    if calibrate_model:
        best_model = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
        best_model.fit(testX, testY)
    model_summary["parameters"] = {}
    model_summary["parameters"].update(best_models_fit_results[measure_name]["shared_params"])
    model_summary["parameters"].update(best_models_fit_results[measure_name]["params"])
    model_summary["combined model performance"] = get_train_test_val_stats_for_model(best_model, trainX, trainY, testX, testY, valX, valY, train_deathY, test_deathY, val_deathY)
    model_summary["death risk stats"] = print_summary_of_death_competing_risk(best_model, testX, testY, valY, train_deathY, test_deathY, val_deathY)
    print_test_stats_for_model(best_model, model_summary["combined model performance"], trainX, trainY, testX, testY,
                patient_feature_vector___index__to__feature__list,
                do_plots=False)
    return model_summary
#%%

stripped_fit_results = [{key: val for key, val in results.items() if key != "combined_model"} for results in fit_results]

train_deathY, test_deathY, val_deathY = get_deathY(subpop_basic_df, trainX_index__to__patient_feature_vector_indexes, testX_index__to__patient_feature_vector_indexes, patient_feature_vector___index__to__lbnr__list)

model_type_summary = {}
best_models_fit_results = get_best_models_fit_results_based_on_training_data(fit_results)

model_type_summary["fit_results"] = stripped_fit_results
#model_type_summary["precision"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "precision", train_deathY, test_deathY, val_deathY)
#model_type_summary["F-0.5"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "F-0.5", train_deathY, test_deathY, val_deathY)
model_type_summary["ROC AUC"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "ROC AUC", train_deathY, test_deathY, val_deathY)
model_type_summary["Precision-Recall Curve AUC"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "Precision-Recall Curve AUC", train_deathY, test_deathY, val_deathY)
model_type_summary["calib ROC AUC"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "ROC AUC", train_deathY, test_deathY, val_deathY, calibrate_model=True)
model_type_summary["calib Precision-Recall Curve AUC"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "Precision-Recall Curve AUC", train_deathY, test_deathY, val_deathY, calibrate_model=True)
#model_type_summary["average precision"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "average precision", train_deathY, test_deathY, val_deathY)
assert(best_model_criterion in model_type_summary)
#model_type_summary["ROC AUC"] = print_and_produce_summary_of_best_model_by_measure(best_models_fit_results, "ROC AUC", train_deathY, test_deathY, val_deathY)

#death_risk_summary = print_summary_of_death_competing_risk(best_models_fit_results["ppv/precision (1000)"]['combined_model'], testX, testY, testX_index__to__patient_feature_vector_indexes, patient_feature_vector___index__to__lbnr__list, subpop_basic_df)

#print("Performance curves for ROC AUC")
#plot_prformance_curves_for_metric(fit_results, "ROC AUC")



#%%
#||
action_name = "Calibrate models and describe their fit results"
action_description = ""
action_input = {"count_trainX", "aux_testX", "aux_valX", "aux_trainX", "count_testX", "count_valX", "trainY", "testY", "valY",
                "blr_fit_results",
                "lr_fit_results",
                "rf_fit_results",
                "xgb_fit_results",
                "patient_feature_vector___index__to__feature__list",
                "testX_index__to__patient_feature_vector_indexes", 
                "patient_feature_vector___index__to__lbnr__list", 
                "subpop_basic_df"}
action_output = {"uncalibrated_model_expected_calibration_error", "calibrated_model_expected_calibration_error"}
#||

#%%

from model_evaluation import multiscorer, multiscorer_confidence_intervals
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

lr_fit_results = load_input(load_input_run_name, "LR_count__fit_results", load_input_shared_storage_name) #|| input
rf_fit_results = load_input(load_input_run_name, "RandFor_count__fit_results", load_input_shared_storage_name) #|| input
blr_fit_results = load_input(load_input_run_name, "LR_aux__fit_results", load_input_shared_storage_name) #|| input
xgb_fit_results = load_input(load_input_run_name, "ExGradientBoost_count__fit_results", load_input_shared_storage_name) #|| input
aux_trainX = load_input(load_input_run_name, "aux__trainX", load_input_shared_storage_name) #|| input
count_trainX = load_input(load_input_run_name, "count__trainX", load_input_shared_storage_name) #|| input
trainY = load_input(load_input_run_name, "trainY", load_input_shared_storage_name) #|| input
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

def get_best_model(fit_results):
    best_models_fit_results = get_best_models_fit_results_based_on_training_data(fit_results)
    return best_models_fit_results[best_model_criterion]['combined_model']

blr_best_model = get_best_model(blr_fit_results)
lr_best_model = get_best_model(lr_fit_results)
rf_best_model = get_best_model(rf_fit_results)
gb_best_model = get_best_model(xgb_fit_results)



blr_test_pred_proba = blr_best_model.predict_proba(aux_testX)[:,1]
lr_test_pred_proba = lr_best_model.predict_proba(count_testX)[:,1]
rf_test_pred_proba = rf_best_model.predict_proba(count_testX)[:,1]
gb_test_pred_proba = gb_best_model.predict_proba(count_testX)[:,1]  

test_score_df = pd.DataFrame(np.hstack((blr_test_pred_proba.reshape((-1,1)),
                                   lr_test_pred_proba.reshape((-1,1)),
                                   rf_test_pred_proba.reshape((-1,1)),
                                   gb_test_pred_proba.reshape((-1,1)),
                                   testY.reshape((-1,1)))),
                        columns=["blr", "lr", "rf", "gb", "y_true"])  



blr_val_pred_proba = blr_best_model.predict_proba(aux_valX)[:,1]
lr_val_pred_proba = lr_best_model.predict_proba(count_valX)[:,1]
rf_val_pred_proba = rf_best_model.predict_proba(count_valX)[:,1]
gb_val_pred_proba = gb_best_model.predict_proba(count_valX)[:,1]

val_score_df = pd.DataFrame(np.hstack((blr_val_pred_proba.reshape((-1,1)),
                                   lr_val_pred_proba.reshape((-1,1)),
                                   rf_val_pred_proba.reshape((-1,1)),
                                   gb_val_pred_proba.reshape((-1,1)),
                                   valY.reshape((-1,1)))),
                        columns=["blr", "lr", "rf", "gb", "y_true"])


#%%

# Get expected calibration error of the uncalibrated classifiers

from math import floor 

def get_model_expected_calibration_error(score_df, nbins=100):
    num_samples = score_df.shape[0]
    bin_size = floor(num_samples/nbins)
    last_bin_size = num_samples%nbins
    
    model_expected_calibration_errors = {"expected_calibration_error": {}, "bins_error": {}, "bin_avg_preds": {}, "bin_case_prop": {}}
    
    for mname in ["blr", "lr", "rf", "gb"]:
        nscore_df = score_df.copy().sort_values(by=mname, axis=0, ascending=True).reset_index(drop=True)
        score_means = nscore_df.groupby(np.floor((nscore_df.index/bin_size).values).astype(np.int)).mean()
        if last_bin_size != 0:
            bin_size_mask = np.ones((nbins + 1))*bin_size/num_samples
            bin_size_mask[-1] = last_bin_size/num_samples
        else:
            bin_size_mask = np.ones((nbins))*bin_size/num_samples
            
        model_expected_calibration_errors["expected_calibration_error"][mname] = sum(np.abs(score_means[mname] - score_means["y_true"]) * bin_size_mask)
        model_expected_calibration_errors["bins_error"][mname] = list(np.abs(score_means[mname] - score_means["y_true"]))
        model_expected_calibration_errors["bin_avg_preds"][mname] = list(score_means[mname])
        model_expected_calibration_errors["bin_case_prop"][mname] = list(score_means["y_true"])
        
    return model_expected_calibration_errors
        
uncalibrated_model_expected_calibration_error = get_model_expected_calibration_error(val_score_df, nbins=100)

#%%

#Calibrate models

#isotonic_calibrator = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
#isotonic_calibrator.fit(testX, testY)
#
#sigmoid_calibrator = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
#sigmoid_calibrator.fit(testX, testY)

blr_calib_best_model = CalibratedClassifierCV(blr_best_model, method="sigmoid", cv="prefit").fit(aux_testX, testY)
lr_calib_best_model = CalibratedClassifierCV(lr_best_model, method="sigmoid", cv="prefit").fit(count_testX, testY)
rf_calib_best_model = CalibratedClassifierCV(rf_best_model, method="sigmoid", cv="prefit").fit(count_testX, testY)
gb_calib_best_model = CalibratedClassifierCV(gb_best_model, method="sigmoid", cv="prefit").fit(count_testX, testY)

calib_blr_val_pred_proba = blr_calib_best_model.predict_proba(aux_valX)[:,1]
calib_lr_val_pred_proba = lr_calib_best_model.predict_proba(count_valX)[:,1]
calib_rf_val_pred_proba = rf_calib_best_model.predict_proba(count_valX)[:,1]
calib_gb_val_pred_proba = gb_calib_best_model.predict_proba(count_valX)[:,1]

calib_val_score_df = pd.DataFrame(np.hstack((calib_blr_val_pred_proba.reshape((-1,1)),
                                   calib_lr_val_pred_proba.reshape((-1,1)),
                                   calib_rf_val_pred_proba.reshape((-1,1)),
                                   calib_gb_val_pred_proba.reshape((-1,1)),
                                   valY.reshape((-1,1)))),
                        columns=["blr", "lr", "rf", "gb", "y_true"])

calibrated_model_expected_calibration_error = get_model_expected_calibration_error(calib_val_score_df, nbins=100)
#%%
plt.plot(uncalibrated_model_expected_calibration_error["bin_avg_preds"]["gb"], uncalibrated_model_expected_calibration_error["bin_case_prop"]["gb"], 's-')
plt.plot(calibrated_model_expected_calibration_error["bin_avg_preds"]["gb"], uncalibrated_model_expected_calibration_error["bin_case_prop"]["gb"], 's-')

#%%

model_labels = {'blr': "Baseline logistic regression", "lr": "Register-based logistic regression", "rf": "Random forest", "gb": "Gradient boosting"}

def plot_model_expected_calibration_error(model_expected_calibration_error, title="Expected calibration error curves"):
    
    for mname, bins in model_expected_calibration_error["bins_error"].items():
        plt.plot(range(len(bins)), bins, 's-', label=model_labels[mname])
    plt.title(title)
    plt.ylabel("Expected calibration error")
    plt.xlabel("Sorted prediction bin")
    plt.legend(loc="upper left")
    plt.show()
    
plot_model_expected_calibration_error(uncalibrated_model_expected_calibration_error, title="Uncalibrated expected calibration error curves")
plot_model_expected_calibration_error(calibrated_model_expected_calibration_error, title="Calibrated expected calibration error curves")

#%%

# Plot calibration curves


def plot_single_model_calibrations(model, trainX=count_trainX, testX=count_testX, valX=count_valX, show_val_fit=False, show_score_distribution=True, n_bins=40):

    isotonic_calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    isotonic_calibrator.fit(testX, testY)

    sigmoid_calibrator = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    sigmoid_calibrator.fit(testX, testY)

    plt.figure(figsize=(10,10))
    prob_pos = isotonic_calibrator.predict_proba(testX)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(testY, prob_pos, n_bins=n_bins)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label="isotonic_calibrator")

    prob_pos = sigmoid_calibrator.predict_proba(testX)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(testY, prob_pos, n_bins=n_bins)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label="sigmoid_calibrator")

    prob_pos = model.predict_proba(testX)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(testY, prob_pos, n_bins=n_bins)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label="non-calibrated")

    if show_score_distribution: 
        plt.plot([(1.0*i)/len(prob_pos) for i in range(len(prob_pos))], sorted(prob_pos), 's-', label="pure_scores")

    plt.ylabel("Fraction of positives")
    plt.xlabel("mean predicted value")
    plt.legend(loc="upper left")
    plt.show()
    
    if show_val_fit: 
        plt.figure(figsize=(10,10))
        prob_pos = isotonic_calibrator.predict_proba(valX)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(valY, prob_pos, n_bins=n_bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label="isotonic_calibrator")
    
        prob_pos = sigmoid_calibrator.predict_proba(valX)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(valY, prob_pos, n_bins=n_bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label="sigmoid_calibrator")
    
        prob_pos = model.predict_proba(valX)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(valY, prob_pos, n_bins=n_bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label="non-calibrated")
    
        if show_score_distribution:
            plt.plot([(1.0*i)/len(prob_pos) for i in range(len(prob_pos))], sorted(prob_pos), 's-', label="pure_scores")
    
        plt.ylabel("Fraction of positives")
        plt.xlabel("mean predicted value")
        plt.legend(loc="upper left")
        plt.show()

#plot_single_model_calibration(blr_best_model, show_val_fit=True, trainX=aux_trainX, testX=aux_testX, valX=aux_valX)
#plot_single_model_calibration(lr_best_model, show_val_fit=True)
#plot_single_model_calibration(rf_best_model, show_val_fit=True)
#plot_single_model_calibrations(gb_best_model, show_val_fit=True, n_bins=100)

#%%

def plot_all_model_calibration(blr_model, lr_model, rf_model, gb_model, testX=count_testX, aux_testX=aux_testX, valX=count_valX, aux_valX=aux_valX, show_val_fit=False, n_bins=50):


    def plot_calibration_curve(model, label, X, Y, n_bins=n_bins):
        prob_pos = model.predict_proba(X)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(Y, prob_pos, n_bins=n_bins)
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=label)

    if not show_val_fit:
        plt.figure(figsize=(10,10))
    
        plot_calibration_curve(blr_model, "Baseline logistic regression", aux_testX, testY)
        plot_calibration_curve(lr_model, "Register logistic regression", testX, testY)
        plot_calibration_curve(rf_model, "Random forest", testX, testY)
        plot_calibration_curve(gb_model, "Gradient boosting", testX, testY)
    
        plt.title("Calibration curves")
        plt.ylabel("Fraction of positives")
        plt.xlabel("mean predicted value")
        plt.legend(loc="upper left")
        plt.show()
        
        
    else: 
        plt.figure(figsize=(10,10))
        
        plot_calibration_curve(blr_model, "Baseline logistic regression", aux_valX, testY)
        plot_calibration_curve(lr_model, "Register logistic regression", valX, testY)
        plot_calibration_curve(rf_model, "Random forest", valX, testY)
        plot_calibration_curve(gb_model, "Gradient boosting", valX, testY)
    
        plt.ylabel("Fraction of positives")
        plt.xlabel("mean predicted value")
        plt.legend(loc="upper left")
        plt.show()

#plot_all_model_calibration(blr_best_model, lr_best_model, rf_best_model, gb_best_model)
#plot_all_model_calibration(blr_calib_best_model, lr_calib_best_model, rf_calib_best_model, gb_calib_best_model)





