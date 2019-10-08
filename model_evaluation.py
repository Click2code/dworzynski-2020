# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:15:55 2017

@author: fskPioDwo
"""

from itertools import product
import numpy as np
from collections import Counter, defaultdict
from pprint import pprint, pformat
import time
import pickle
from os import path
from multiprocessing import Pool, current_process
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
import math
import re

from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, fbeta_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score

from auroc_CIs import delong_roc_variance
from scipy.stats import norm

from pprint import pprint


class Fold:
    def __init__(self, X, Y, train_index, test_index):
        self.trainXfold = X[train_index]
        self.trainYfold = Y[train_index]
        self.testXfold = X[test_index]
        self.testYfold = Y[test_index]

#
#This is my own implementation
#
class MyOwnGridSearchCV:

    kf = None
    folds = None
    results = None
    fold_indexes = None



    def __init__(self, estimator_class, num_k_folds, param_grid, shared_params, score_func, fold_indexes):
        self.estimator_class = estimator_class
        self.num_k_folds = num_k_folds
        self.param_grid = param_grid
        self.shared_params = shared_params
        self.score_func = score_func
        self.fold_indexes = fold_indexes #if fold_indexes != None else KFold(n=n_samples, n_folds=self.num_k_folds, shuffle=True, random_state=0)

        annotated_param_values = [[(param_name, param_value) for param_value in param_values] for param_name, param_values in list(param_grid.items()) ]
        self.param_sets = list(product(*annotated_param_values))

        self.total_num_params = len(self.param_sets)*(self.num_k_folds+1)
        print(("Total number of parameter sets: {} a total number of {} models will be trained".format(len(self.param_sets), self.total_num_params)))

    @staticmethod
    def train_model_on_parameter_set(param_set, X_file_path, Y, shared_params, fold_indexes, num_k_folds, estimator_class, score_func):

        #X = np.concatenate(X_dilled)

        X = pickle.load(open(X_file_path, "rb"))

        print(("Training param set {}:\n\t{}".format(param_set, "\n\t".join(["{}: {}".format(param, val) for param, val in param_set]))))
        param_set_models = []
        param_set_combined_model = None
        param_set_scores_list = []

        combined_model_params = {}
        combined_model_params.update(shared_params)
        combined_model_params.update({k: v for k, v in param_set})

        additional_fit_params = {}
        if "additional_fit_params" in combined_model_params:
            additional_fit_params = combined_model_params["additional_fit_params"]
            del combined_model_params["additional_fit_params"]

        insert_xgb_eval_set = False
        if "insert_xgb_eval_set" in combined_model_params and combined_model_params["insert_xgb_eval_set"] is True:
            del combined_model_params["insert_xgb_eval_set"]
            insert_xgb_eval_set = True

        best_num_iters = []

        #Prepare K-folds
        folds = [Fold(X, Y, train_index, test_index) for train_index, test_index in fold_indexes]#This will store everything in RAM, probably not best idea

        for fold_i, fold in enumerate(folds):
            start_time = time.time()
            print(("Training Fold {}/{}".format(fold_i+1, num_k_folds)))
            model = estimator_class(**combined_model_params)

            if insert_xgb_eval_set:
                #additional_fit_params["eval_set"] = [(fold.testXfold, fold.testYfold)]
                model.fit(fold.trainXfold, fold.trainYfold, eval_set=[(fold.testXfold, fold.testYfold)], **additional_fit_params)
                if hasattr(model, "best_ntree_limit"):
                    best_num_iters.append(model.best_ntree_limit)
                else:
                    best_num_iters.append(combined_model_params["n_estimators"])
            else:
                model.fit(fold.trainXfold, fold.trainYfold, **additional_fit_params)
            param_set_models.append(model)
            #fold_score = score_func(model, fold.testXfold, fold.testYfold)

            fold_pred_proba = model.predict_proba(fold.testXfold)[:,1]
            fold_pred_bin = (fold_pred_proba > 0.5).astype(np.int)
            fold_score_df = asssemble_score_dataframe(fold_pred_proba, fold_pred_bin, fold.testYfold, died=None)
            fold_score = score_func(fold_score_df)
            param_set_scores_list.append(fold_score)
            print(("Done in: {}".format(time.time() - start_time)))

        print("Training a combined model on full training set")
        if isinstance(model, XGBClassifier) and hasattr(model, "best_ntree_limit"):
            avg_number_of_best_iters = int(sum(best_num_iters)/len(best_num_iters))
            print(f"Average number of best iterations: {avg_number_of_best_iters}")
            combined_model_params["n_estimators"] = avg_number_of_best_iters
            param_set_combined_model = estimator_class(**combined_model_params)
            param_set_combined_model.fit(X,Y, verbose=1)
        else:
            param_set_combined_model = estimator_class(**combined_model_params)
            param_set_combined_model.fit(X,Y)

        #Combine scores from each individual fold into one averaged score dict
        param_set_scores = defaultdict(lambda: 0)

        for scores in param_set_scores_list:
            for score, score_val in list(scores.items()):
                if isinstance(score_val, (int, float)): #this is silly but a result of adding CIs to ROC AUC calculations - we probably don't need to return scores here
                    param_set_scores[score] += score_val
        for score, score_val in list(param_set_scores.items()):
            param_set_scores[score] /= (len(param_set_scores_list) * 1.0)

        return {
                "combined_model": param_set_combined_model,
    #                "models": param_set_models, # individual models aren't saved
                "shared_params": shared_params,
                "avg_scores": dict(param_set_scores),
                "params": param_set
                }

    def load_tmp_results(self, tmp_file_path):
        if tmp_file_path == None or not path.exists(tmp_file_path):
            print("No previous training file found, will start from scratch")
            results = []
        else:
            print("Loading previous training file")
            with open(tmp_file_path, mode="rb") as f:
                results = pickle.load(f)
            print(f"Previous training file loaded. Number of param_sets trained: {len(results)}")
        return results

    def single_process_fit(self, X, Y, tmp_dir_path, pipeline_action_prefix):

        print("Starting training")

        tmp_file_path = path.join(tmp_dir_path, pipeline_action_prefix + "__fit_results_tmp.pickle")

        results = self.load_tmp_results(tmp_file_path)

        X_file_path = path.join(tmp_dir_path, pipeline_action_prefix + "__X_tmp.pickle")
        pickle.dump(X, open(X_file_path, "wb"), protocol=-1)

        for param_set_i, param_set in enumerate(self.param_sets):
            if param_set in {result["params"] for result in results}:
                print(f"{param_set} was already trained, skipping")
                continue
            print(f"Starting run for {param_set}")
            kwds={"param_set": param_set, "X_file_path": X_file_path, "Y": Y, "shared_params": self.shared_params,
                  "fold_indexes": self.fold_indexes, "num_k_folds": self.num_k_folds, "estimator_class": self.estimator_class,
                  "score_func": self.score_func}

            result = MyOwnGridSearchCV.train_model_on_parameter_set(**kwds)
            print("got results for a parameter set: {}".format(result["params"]))
            results.append(result)

            if tmp_file_path:
                with open(tmp_file_path, mode="wb") as f:
                    pickle.dump(results, f)

        return results

    def parallel_fit(self, X, Y, tmp_dir_path, pipeline_action_prefix, num_processes=5):

        def parameterset_result_ready(results, new_result):
            results.append(new_result)
            print("got results for a parameter set: {}".format(new_result["params"]))

        print("Starting training")

        tmp_file_path = path.join(tmp_dir_path, pipeline_action_prefix + "__fit_results_tmp.pickle")

        results = self.load_tmp_results(tmp_file_path)

        X_file_path = path.join(tmp_dir_path, pipeline_action_prefix + "__X_tmp.pickle")
        pickle.dump(X, open(X_file_path, "wb"), protocol=-1)

        print(f"Spawning pool of {num_processes} processes")
        process_pool = Pool(min(num_processes, len(self.param_sets)), maxtasksperchild=1)
        for param_set_i, param_set in enumerate(self.param_sets):
            if param_set in {result["params"] for result in results}:
                print(f"{param_set} was already trained, skipping")
                continue
            print(f"Scheduling run for {param_set}")

            process_pool.apply_async(func=MyOwnGridSearchCV.train_model_on_parameter_set,
                                     args=(),
                                     kwds={"param_set": param_set, "X_file_path": X_file_path, "Y": Y,
                                           "shared_params": self.shared_params, "fold_indexes": self.fold_indexes, "num_k_folds": self.num_k_folds, "estimator_class": self.estimator_class, "score_func": self.score_func},
                                     #callback=lambda new_result: save_tmp_results(new_result, tmp_file_path), #This could have been causing issues with parallelization as was running too long
                                     callback=lambda new_result: parameterset_result_ready(results, new_result),
                                     error_callback=lambda exc: print("Training for parameter set {} failed, error:\n{}".format(param_set, exc))
                                     )

        process_pool.close() #Prevents more tasks from being added to the pool, worksers will exit once all tasks are completed

        #Monitor results variable for updates and save those on the fly
        old_results_lentgh = len(results)
        while True:
            time.sleep(5)# seconds
            new_results_length = len(results)
            if new_results_length != old_results_lentgh:
                print(f"{new_results_length - old_results_lentgh} new results observed, saving")
                if tmp_file_path:
                    with open(tmp_file_path, mode="wb") as f:
                        pickle.dump(results, f)

                old_results_lentgh = len(results)

            if new_results_length == len(self.param_sets) or old_results_lentgh >= len(self.param_sets):
                break

        process_pool.join()

        #self.results = load_tmp_results(tmp_file_path)
        return results

    def fit(self, X, Y, tmp_file_path=None):
        #Prepare K-folds
        self.folds = [Fold(X, Y, train_index, test_index) for train_index, test_index in self.fold_indexes]#This will store everything in RAM, probably not best idea
        #scikit 0.18
        #self.kf = KFold(n_splits=self.num_k_folds, random_state=0)
        #self.folds = [self.Fold(X, Y, train_index, test_index) for train_index, test_index in self.kf.split(trainX)]#This will store everything in RAM, probably not best idea

        print("Starting training")

        if tmp_file_path == None or not path.exists(tmp_file_path):
            print("No previous training file found, will start from scratch")
            results = []
        else:
            print("Loading previous training file")
            with open(tmp_file_path, mode="rb") as f:
                results = pickle.load(f)
                print(f"Previous training file loaded. Number of param_sets trained: {len(results)}")

        for param_set_i, param_set in enumerate(self.param_sets):

            if param_set in {result["params"] for result in results}:
                print(f"Oh, {param_set} was already trained, skipping")
                continue

            print(("Training param set #{}/{}:\n\t{}".format(param_set_i, len(self.param_sets), "\n\t".join(["{}: {}".format(param, val) for param, val in param_set]))))
            param_set_models = []
            param_set_combined_model = None
            param_set_scores_list = []

            combined_model_params = {}
            combined_model_params.update(self.shared_params)
            combined_model_params.update({k: v for k, v in param_set})

            for fold_i, fold in enumerate(self.folds):
                start_time = time.time()
                print(("Training Fold {}/{}".format(fold_i+1, self.num_k_folds)))
                model = self.estimator_class(**combined_model_params)
                model.fit(fold.trainXfold, fold.trainYfold)
                param_set_models.append(model)
                fold_score = self.score_func(model, fold.testXfold, fold.testYfold)
                param_set_scores_list.append(fold_score)
                print(("Done in: {}".format(time.time() - start_time)))
            print("Training a combined model on full training set")
            param_set_combined_model = self.estimator_class(**combined_model_params)
            param_set_combined_model.fit(X,Y)

            #Combine scores from each individual fold into one averaged score dict
            param_set_scores = defaultdict(lambda: 0)
            for scores in param_set_scores_list:
                for score, score_val in list(scores.items()):
                    if isinstance(score_val) == tuple:
                        score_val = score_val[0]
                    param_set_scores[score] += score_val
            for score, score_val in list(param_set_scores.items()):
                param_set_scores[score] /= (len(param_set_scores_list) * 1.0)

            results.append({
                "param_set_i": param_set_i,
                "combined_model": param_set_combined_model,
#                "models": param_set_models, # individual models aren't saved
                "shared_params": self.shared_params,
                "avg_scores": dict(param_set_scores),
                "params": param_set
                })

            if tmp_file_path:
                with open(tmp_file_path, mode="wb") as f:
                    pickle.dump(results, f)

            #pipeline_force_save_obj(results, fit_results)

        self.results = results
        return results

def asssemble_score_dataframe(y_pred_prob, y_pred_bin, y_true, died=None):
    if died is not None:
        d = pd.DataFrame(np.hstack((y_pred_prob.reshape((-1,1)), y_pred_bin.reshape((-1,1)), y_true.reshape((-1,1)), died.reshape((-1,1)))), columns=["pred_outcome_prob", "pred_outcome_bin", "y_true", "died"])
        d["died"] = d["died"].astype(np.int)
    else:
        d = pd.DataFrame(np.hstack((y_pred_prob.reshape((-1,1)), y_pred_bin.reshape((-1,1)), y_true.reshape((-1,1)))), columns=["pred_outcome_prob", "pred_outcome_bin", "y_true"])
    d["y_true"] = d["y_true"].astype(np.int)
    d["rank"] = d[["pred_outcome_prob"]].rank(ascending=False, method="min").astype(np.int) #method must be min to account for multiple patients with the same score
    d["pred_outcome_bin"] = d["pred_outcome_bin"].astype(np.int)

    d.sort_values(by="pred_outcome_prob", ascending=False, inplace=True)

    return d

def binarize_scores(score_df, score_threshold, use_only_top_X_predictions_X=None):
    binary_scores = (score_df["pred_outcome_prob"] > score_threshold).astype(np.int)
    if use_only_top_X_predictions_X != None:
        binary_scores.loc[score_df["rank"] > use_only_top_X_predictions_X] = 0

    return binary_scores

#This will correctly set the binary y_pred vector based on both: rank (through x) and score>0.5
def use_only_top_X_predictions(y_pred_binary, y_pred_prob, x, precalc_df=None, score_threshold=0.5):
    if precalc_df == None:
        d = pd.DataFrame(np.hstack((y_pred_prob.reshape((-1,1)), y_pred_binary.reshape((-1,1)))), columns=["pred_outcome_prob", "pred_outcome_bin"])
        d["rank"] = d[["pred_outcome_prob"]].rank(ascending=False, method="min") #method must be min to account for multiple patients with the same score
    #d["pred_outcome_bin"] = d["pred_outcome_bin"].astype(np.int)
    d["pred_outcome_bin"] = (d["pred_outcome_prob"] > score_threshold).astype(np.int)
    d.loc[d["rank"] > x, "pred_outcome_bin"] = 0

    return d["pred_outcome_bin"]

#specificity = true_negatives/negatives
def specificity_score(score_df, use_only_top_X_predictions_X=None):
    pred_bin = binarize_scores(score_df, 0.5, use_only_top_X_predictions_X)
    true_labels = score_df["y_true"]

    def count_zeros(arr):
        return arr.shape[0] - np.count_nonzero(arr)

    num_negatives = count_zeros(true_labels)
    num_true_negatives = count_zeros(true_labels | pred_bin)

    if num_negatives == 0:
        return 0

    specificity = num_true_negatives/num_negatives

    return specificity

#sensitivity = true_positives/positives
def sensitivity_score(score_df, use_only_top_X_predictions_X=None):
    pred_bin = binarize_scores(score_df, 0.5, use_only_top_X_predictions_X)
    true_labels = score_df["y_true"]

    num_positives = np.count_nonzero(true_labels)
    num_true_positives = np.count_nonzero(true_labels & pred_bin)

    if num_positives == 0:
        return 0

    sensitivity = num_true_positives/num_positives

    return sensitivity


def auroc_CIs(y_true, y_score, confidence = 0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_score)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - confidence)/2.0)

    ci = norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

    return (ci[0], ci[1])

def average_precision_score_CI(y_true, y_pred_score, y_pred_binary):
    indexes = np.arange(0, y_pred_score.shape[0], 1)

    avg_precision_sample_results = np.zeros(1000)
    for sample_i in range(1000):
        #print(sample_i)
        sample_indexes = np.random.choice(indexes, size=indexes.shape, replace=True) #sampling with replaces
        sample_y_pred_prob = y_pred_score[sample_indexes]
        #sample_y_pred_binary = y_pred_binary[sample_indexes]
        sample_Y = y_true[sample_indexes]
        if np.sum(sample_Y) < 1:
            sample_i -= 1
            continue

        avg_precision_sample_results[sample_i] = average_precision_score(y_true = sample_Y, y_score=sample_y_pred_prob, average="weighted")

    avg_precision_sample_results = np.sort(avg_precision_sample_results)

    return (avg_precision_sample_results[25], avg_precision_sample_results[975])


def incidence(y_true):
    return sum(y_true)/len(y_true)


def multiscorer(score_df): #"pred_outcome_prob", "pred_outcome_bin", "y_true"
    y_true = score_df["y_true"]
    y_pred = score_df["pred_outcome_bin"]
    y_score = score_df["pred_outcome_prob"]
    return {
        "Accuracy": accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True),
        #"average precision": average_precision_score(y_true=Y, y_score=y_pred_prob, average="weighted"),
        "ROC AUC": roc_auc_score(y_true=y_true, y_score=y_score),
        "ROC AUC CI": auroc_CIs(y_true=y_true, y_score=y_score),
        "Precision-Recall Curve AUC": average_precision_score(y_true=y_true, y_score=y_score, average="weighted"),
        "Precision-Recall Curve AUC CI": average_precision_score_CI(y_true=y_true, y_pred_score=y_score, y_pred_binary=y_pred),
        "F-0.1": fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.1, average="binary", pos_label=1),
        "F-0.5": fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.5, average="binary", pos_label=1),
        "F-1": f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1),
        "F-2": fbeta_score(y_true=y_true, y_pred=y_pred, beta=2, average="binary", pos_label=1),
        "precision": precision_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1),
        "recall": recall_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1),
        "specificity (ALL)": specificity_score(score_df, use_only_top_X_predictions_X=None),
        "specificity (100)": specificity_score(score_df, use_only_top_X_predictions_X=100),
        "specificity (500)": specificity_score(score_df, use_only_top_X_predictions_X=500),
        "specificity (1000)": specificity_score(score_df, use_only_top_X_predictions_X=1000),
        "specificity (5000)": specificity_score(score_df, use_only_top_X_predictions_X=5000),
        #"specificity (10000)": specificity_score(y_true=Y, y_pred=y_pred, y_pred_prob=y_pred_prob, use_only_top_X_predictions_X=10000),
        "sensitivity/recall (ALL)": sensitivity_score(score_df, use_only_top_X_predictions_X=None),
        "sensitivity/recall (100)": sensitivity_score(score_df, use_only_top_X_predictions_X=100),
        "sensitivity/recall (500)": sensitivity_score(score_df, use_only_top_X_predictions_X=500),
        "sensitivity/recall (1000)": sensitivity_score(score_df, use_only_top_X_predictions_X=1000),
        "sensitivity/recall (5000)": sensitivity_score(score_df, use_only_top_X_predictions_X=5000),
        #"sensitivity/recall (10000)": sensitivity_score(y_true=Y, y_pred_binary=y_pred, y_pred_prob=y_pred_prob, use_only_top_X_predictions_X=10000),
        "True Incidence": incidence(y_true=y_true),
        "ppv/precision (ALL)": ppv_score(score_df, use_only_top_X_predictions_X=None)[0],
        "ppv/precision (ALL) SE": ppv_score(score_df, use_only_top_X_predictions_X=None)[1],
        "ppv/precision (100)": ppv_score(score_df, use_only_top_X_predictions_X=100)[0],
        "ppv/precision (500)": ppv_score(score_df, use_only_top_X_predictions_X=500)[0],
        "ppv/precision (1000)": ppv_score(score_df, use_only_top_X_predictions_X=1000)[0],
        "ppv/precision (5000)": ppv_score(score_df, use_only_top_X_predictions_X=5000)[0],
        #"ppv/precision (10000)": ppv_score(y_true=Y, y_pred_binary=y_pred, y_pred_prob=y_pred_prob, use_only_top_X_predictions_X=10000),
        }

def multiscorer_confidence_intervals(y_pred_prob, y_pred_binary, Y):
    indexes = np.arange(0, y_pred_prob.shape[0], 1)

    roc_auc_sample_results = np.zeros(1000)
    ppv1000_sample_results = np.zeros(1000)
    avg_precision_sample_results = np.zeros(1000)

    for sample_i in range(1000):
        #print(sample_i)
        sample_indexes = np.random.choice(indexes, size=indexes.shape, replace=True) #sampling with replaces
        sample_y_pred_prob = y_pred_prob[sample_indexes]
        sample_y_pred_binary = y_pred_binary[sample_indexes]
        sample_Y = Y[sample_indexes]
        if np.sum(sample_Y) < 1:
            sample_i -= 1
            continue

        roc_auc_sample_results[sample_i] = roc_auc_score(y_true = sample_Y, y_score=sample_y_pred_prob)
        avg_precision_sample_results[sample_i] = average_precision_score(y_true = sample_Y, y_score=sample_y_pred_prob, average="weighted")
        ppv, ppv_SE = ppv_score(y_true=sample_Y, y_pred_binary=sample_y_pred_binary, y_pred_prob=sample_y_pred_prob, use_only_top_X_predictions_X=1000)
        ppv1000_sample_results[sample_i] = ppv

    roc_auc_sample_results = np.sort(roc_auc_sample_results)
    ppv1000_sample_results = np.sort(ppv1000_sample_results)
    avg_precision_sample_results = np.sort(avg_precision_sample_results)

    confidence_intervals = {
            "Precision-Recall Curve AUC": {
                    "mid": avg_precision_sample_results[500],
                    "upper": avg_precision_sample_results[975],
                    "lower": avg_precision_sample_results[25]
                    },
            "ROC AUC": {
                    "mid": roc_auc_sample_results[500],
                    "upper": roc_auc_sample_results[975],
                    "lower": roc_auc_sample_results[25]
                    },
            "ppv/precision (1000)": {
                    "mid": ppv1000_sample_results[500],
                    "upper": ppv1000_sample_results[975],
                    "lower": ppv1000_sample_results[25]
                    }
            }

    return confidence_intervals

#ppv = true_positives/(true_positives + false_positives)
def ppv_score(score_df, score_threshold=0.5, use_only_top_X_predictions_X=None):
    pred_bin = binarize_scores(score_df, score_threshold, use_only_top_X_predictions_X)
    true_labels = score_df["y_true"]

    num_true_positives = np.count_nonzero(true_labels & pred_bin)
    num_false_positives = np.count_nonzero(pred_bin - (pred_bin & true_labels))
    num_pred_positives = np.count_nonzero(pred_bin)

    if num_pred_positives == 0:
        return 0, 0

    #ppv = num_true_positives/(num_true_positives + num_false_positives)
    ppv = num_true_positives/num_pred_positives


    ppv_SE_1 = num_true_positives/num_pred_positives
    ppv_SE_2 = num_false_positives/num_pred_positives
    ppv_SE = math.sqrt(ppv_SE_1 * ppv_SE_2 / num_pred_positives)

    return ppv, ppv_SE


def calculate_incidence_at_percentile(score_df):
    incicence_at_percentile = {}

    sscore_df = score_df.sort_values(by="pred_outcome_prob", axis=0, ascending=True, inplace=False)
    num_individuals_in_percentile = math.floor(sscore_df.shape[0]/100.0)

    for p in range(0,100):
        p_i = p*num_individuals_in_percentile
        incidence = sscore_df.iloc[p_i:p_i + num_individuals_in_percentile]["y_true"].sum()
        incicence_at_percentile[p] = (incidence*1.0)/num_individuals_in_percentile
        #print(f"{p}\t{incidence}")

    return incicence_at_percentile

def calculate_death_incidence_at_percentile(score_df):
    incicence_at_percentile = {}

    sscore_df = score_df.sort_values(by="pred_outcome_prob", axis=0, ascending=True, inplace=False)
    num_individuals_in_percentile = math.floor(sscore_df.shape[0]/100.0)

    for p in range(0,100):
        p_i = p*num_individuals_in_percentile
        incidence = sscore_df.iloc[p_i:p_i + num_individuals_in_percentile]["died"].sum()
        incicence_at_percentile[p] = (incidence*1.0)/num_individuals_in_percentile
        #print(f"{p}\t{incidence}")

    return incicence_at_percentile

#def calculate_incidence_at_permile(score_df):
#    incicence_at_permile = {}
#
#    sscore_df = score_df.sort_values(by="pred_outcome_prob", axis=0, ascending=True, inplace=False)
#    num_individuals_in_permile = math.floor(sscore_df.shape[0]/1000.0)
#
#    for p in range(0,1000):
#        p_i = p*num_individuals_in_permile
#
#        incidence = sscore_df.iloc[p_i:p_i + num_individuals_in_permile]["y_true"].sum()
#        incicence_at_permile[p] = (incidence*1.0)/num_individuals_in_permile
#        #print(f"{p}\t{incidence}")
#
#    return incicence_at_permile


def calculate_ppv_at_X(score_df, ppv_at_X_range, best_score_thresholds_atK_in_test=0.5):#y_pred_prob, y_pred_binary, y_true, deathY):
    ppv_at_X_agg = {}

    #d = pd.DataFrame(np.hstack((y_pred_prob.reshape((-1,1)), y_pred_binary.reshape((-1, 1)), deathY.reshape((-1,1)))), columns=["pred_outcome_prob", "pred_outcome_bin", "died"])
    #d["rank"] = d[["pred_outcome_prob"]].rank(ascending=False, method="min") #method must be min to account for multiple patients with the same score
    #sorted_outcome_probs = sorted(list(d["pred_outcome_prob"]), reverse=True)

    #max_x = 10000
    #num_pred_positives = sum(y_pred_binary)
    #top_x = min(num_pred_positives, max_x)
    #for x in np.linspace(1, top_x, int(top_x/100)):
    for x in ppv_at_X_range:
        if isinstance(best_score_thresholds_atK_in_test, (float)):
            ppv_at_X, ppv_at_X_SE = ppv_score(score_df, best_score_thresholds_atK_in_test, use_only_top_X_predictions_X=x)
        else:
            ppv_at_X, ppv_at_X_SE = ppv_score(score_df, best_score_thresholds_atK_in_test[x], use_only_top_X_predictions_X=x)

        #This is using just ranks - it is fair but X becomes rank and might not represent individuals well (topX loses meaning)
#        num_dead = score_df.loc[score_df["rank"] < x, "died"].sum()
#        pred_risk = score_df.loc[score_df["rank"] == x, "pred_outcome_prob"].mean()
#        num_individuals_with_or_below_given_rank = len(score_df[score_df["rank"] < x])  #this is to account for multiple patients with same score
#        prop_dead_at_X = num_dead/num_individuals_with_or_below_given_rank

        #This is using just ranks - it is fair but X becomes rank and might not represent individuals well (topX loses meaning)
        num_dead = score_df["died"].iloc[:x].sum()
        pred_risk = score_df["pred_outcome_prob"].iloc[x].mean()
        prop_dead_at_X = num_dead/float(x)

        #ppv_at_X_agg[int(x)] = {"ppv": ppv_at_X, "drr": prop_dead_at_X, "pred_risk": sorted_outcome_probs[int(x)], "ppv_SE": ppv_at_X_SE}
        ppv_at_X_agg[int(x)] = {"ppv": ppv_at_X, "drr": prop_dead_at_X, "pred_risk": pred_risk, "ppv_SE": ppv_at_X_SE}

    return ppv_at_X_agg


def get_every_Xth_in_3long_tuple(X, t):
    return (t[0][::X], t[1][::X])

#def get_best_score_threshold_at_K_for_f05(ppv_at_X_range, test_score_df):
def get_best_score_threshold_at_K_for_a_metric(ppv_at_X_range, test_score_df, metric):
    sorted_scores = sorted(test_score_df["pred_outcome_prob"], reverse=False)
    rounded_sorted_scores = np.around(sorted_scores, decimals=2)

    best_score_threshold_at_K = {}
    for k in ppv_at_X_range:
        best_score_threshold = None
        #best_fbeta = 0
        best_metric_score = 0
        for score_threshold in np.unique(rounded_sorted_scores[:k]):
            pred_bin = binarize_scores(test_score_df, score_threshold, use_only_top_X_predictions_X=k)
            #fbeta = fbeta_score(y_true = test_score_df["y_true"], y_pred=pred_bin, beta=0.5, average="binary", pos_label=1)
            metric_score = metric(test_score_df["y_true"], pred_bin)
            #if fbeta > best_fbeta:
            if metric_score > best_metric_score:
                best_score_threshold = score_threshold
                #best_fbeta = fbeta
                best_metric_score = metric_score
        best_score_threshold_at_K[k] = best_score_threshold

    return best_score_threshold_at_K


def get_model_perf_summary(score_df, ppv_at_X_range, make_full_summary=False, best_score_thresholds_atK_in_test=None):


    perf_summary = {
            "classification_report": classification_report(score_df["y_true"], score_df["pred_outcome_bin"], target_names=["controls", "cases"]),
            "confusion_matrix": confusion_matrix(score_df["y_true"], score_df["pred_outcome_bin"]),
            "metrics": multiscorer(score_df),
    }

    if make_full_summary:
        perf_summary.update({
            "roc_curve": get_every_Xth_in_3long_tuple(5, roc_curve(score_df["y_true"], score_df.pred_outcome_prob, 1)),
            "precision_recall_curve": get_every_Xth_in_3long_tuple(5, precision_recall_curve(score_df["y_true"], score_df.pred_outcome_prob, 1)),
            "ppv_at_X": calculate_ppv_at_X(score_df, ppv_at_X_range, best_score_thresholds_atK_in_test),
            "agg_inci_at_X": calculate_ppv_at_X(score_df, ppv_at_X_range, 0.0),
            "incidence_at_percentile": calculate_incidence_at_percentile(score_df),
            "death_incidence_at_percentile": calculate_death_incidence_at_percentile(score_df),
            #"confidence_intervals": multiscorer_confidence_intervals(y_pred_prob, y_pred_binary, Y)
        })
    return perf_summary


def get_train_test_val_stats_for_model(model, trainX, trainY, testX, testY, valX, valY, train_deathY, test_deathY, val_deathY):

    #ppv_at_X_range = range(100, 1000 + 100, 100)
    #ppv_at_X_range_f = lambda score_df: range(100, math.ceil(score_df.shape[0]/100.0 + 100), 100) #This is the top percentile, because test and val are small the percentile is always around 300 individuals
    ppv_at_X_range_f = lambda score_df: range(100, score_df.shape[0], 100)

    assemble_scores = lambda X, Y, D: asssemble_score_dataframe(model.predict_proba(X)[:,1], model.predict(X), Y, died=D)
    train_scores_df = assemble_scores(trainX, trainY, train_deathY)
    test_scores_df = assemble_scores(testX, testY, test_deathY)
    val_scores_df = assemble_scores(valX, valY, val_deathY)

    best_score_thresholds_atK_in_test = get_best_score_threshold_at_K_for_a_metric(ppv_at_X_range_f(val_scores_df), test_scores_df, metric=lambda yt, yp: fbeta_score(y_true=yt, y_pred=yp, beta=0.5, average="binary", pos_label=1))

    train_model_perf_summary = get_model_perf_summary(train_scores_df, ppv_at_X_range_f(train_scores_df), make_full_summary=False)
    test_model_perf_summary = get_model_perf_summary(test_scores_df, ppv_at_X_range_f(test_scores_df), make_full_summary=False)
    val_model_perf_summary = get_model_perf_summary(val_scores_df, ppv_at_X_range_f(val_scores_df), make_full_summary=True, best_score_thresholds_atK_in_test=best_score_thresholds_atK_in_test)

    return {
            "train": train_model_perf_summary,
            "test": test_model_perf_summary,
            "val": val_model_perf_summary
    }


def print_test_stats_for_model(model, model_perf_summary, trainX, trainY, testX, testY, patient_feature_vector___index__to__feature__list, do_plots=False):

    print("\n\n#\n#Test Classification Report\n#")
    print(model_perf_summary["test"]["classification_report"])

    print("\n\n#\n#Test Confusion Matrix\n#")
    print("Y - True Label; X - Predicted Label")
    print(model_perf_summary["test"]["confusion_matrix"])

    print("\n\n#\n#Test Metrics\n#")
    pprint(model_perf_summary["test"]["metrics"])

    do_feature_importances = hasattr(model, "feature_importances_") or hasattr(model, "coef_")

    if do_feature_importances:
        print("\n\n#\n#Feature importances\n#")
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            if isinstance(model, (XGBClassifier, GradientBoostingClassifier)):
                feature_importances_std = np.zeros_like(feature_importances)
            else:
                feature_importances_std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        elif hasattr(model, "coef_"):
#            print("model.coef_")
#            print(len(model.coef_))
#            print(model.coef_)
#            print(model.coef_[0])
            feature_importances = np.abs(model.coef_[0])
            feature_importances_std = np.zeros_like(feature_importances)
        else:
            raise Exception("This is pretty model specific, if it fails look here")

        feature_importances_indices = np.argsort(feature_importances)[::-1]

        print("Most important features:")

        #Collect all feature importances
        #Sometimes a feature is categorical and one-hot-encoded thus its iportance is distributed across multiple features and needs to be summed
        feature_importances_dict = defaultdict(lambda : 0)
        for i in range(min(500,len(feature_importances_indices))):
#            print("i") #del
#            print(i) #del
#            print("feature_importances_indices[i]") # del
#            print(feature_importances_indices[i]) # del
#            print("feature_importances[feature_importances_indices[i]]") #del
#            print(feature_importances[feature_importances_indices[i]]) #del
#            print("patient_feature_vector___index__to__feature__list[feature_importances_indices[i]]") #del
#            print(patient_feature_vector___index__to__feature__list[feature_importances_indices[i]]) #del

            feature_name = patient_feature_vector___index__to__feature__list[feature_importances_indices[i]]

            #Some features are in one-hot encoding and their importances need to be summed, see if current one is like that
            match_for_categorical =  re.match("C\((.*)\)\[.*\]", feature_name)
            if match_for_categorical:
                feature_name = match_for_categorical.groups()[0]

            feature_importance = feature_importances[feature_importances_indices[i]]
            feature_importances_dict[feature_name] += feature_importance

        model_perf_summary["test"]["feature_importances"] = sorted(feature_importances_dict.items(), key=lambda fname_importance: fname_importance[1], reverse=True)

        for i in range(min(10,len(model_perf_summary["test"]["feature_importances"]))):
            print(("feature {}: {}".format(model_perf_summary["test"]["feature_importances"][i][0], model_perf_summary["test"]["feature_importances"][i][1])))

        #print("Worst Features:")
        #for i in range(len(feature_importances_indices)-10,len(feature_importances_indices)):
        #    print("feature {}: {}".format(patient_feature_vector___index__to__feature__list[feature_importances_indices[i]], feature_importances[feature_importances_indices[i]]))

        def get_boxplot_stats(values):
            #print(values)
            #print("abrakadabra")
            upper_quartile = np.percentile(values, 75)
            lower_quartile = np.percentile(values, 25)
            #print(f"upper_quartile: {upper_quartile}")
            #print(f"lower_quartile: {lower_quartile}")
            #print(f"values.shape: {values.shape}")
            iqr =  upper_quartile - lower_quartile

            return {
                "median" : np.median(values),
                "upper_quartile" : upper_quartile,
                "lower_quartile" : lower_quartile,
                "upper_whisker" : values[values <= upper_quartile+1.5*iqr].max(),
                "lower_whisker" : values[values >= lower_quartile-1.5*iqr].min(),
                "count_nonzero" : np.count_nonzero(values),
                "count_total": len(values)
                }

        predicted_probabilities = model.predict_proba(testX)
        predicted_probabilities_sorted_indexes = np.argsort(predicted_probabilities[:, 1])
#        top1000_individuals_indexes = predicted_probabilities_sorted_indexes[-1000:] #note that these are sorted in ascending fasion
#        rest_individuals_indexes = predicted_probabilities_sorted_indexes[:-1000]
#        model_perf_summary["test"]["important_feature_boxplots__top1000"] = []
#        model_perf_summary["test"]["important_feature_boxplots__rest"] = []
#        for i in range(min(10, len(feature_importances_indices))):
#            feature_name = patient_feature_vector___index__to__feature__list[feature_importances_indices[i]]
#
#            top1000_boxplot_stats = get_boxplot_stats(trainX[top1000_individuals_indexes, feature_importances_indices[i]])
#            rest_boxplot_stats = get_boxplot_stats(trainX[rest_individuals_indexes, feature_importances_indices[i]])
#
#            model_perf_summary["test"]["important_feature_boxplots__top1000"].append((i, feature_name, top1000_boxplot_stats))
#            model_perf_summary["test"]["important_feature_boxplots__rest"].append((i, feature_name, rest_boxplot_stats))

        num_individuals_in_20_percentiles = math.ceil(len(predicted_probabilities_sorted_indexes)/20.0)
        top20percentile_individuals_indexes = predicted_probabilities_sorted_indexes[-num_individuals_in_20_percentiles:] #note that these are sorted in ascending fasion
        bottom20percentiles_individuals_indexes = predicted_probabilities_sorted_indexes[:num_individuals_in_20_percentiles]
        model_perf_summary["test"]["important_feature_boxplots__top20percentiles"] = []
        model_perf_summary["test"]["important_feature_boxplots__bottom20percentiles"] = []
        for i in range(min(10, len(feature_importances_indices))):
            feature_name = patient_feature_vector___index__to__feature__list[feature_importances_indices[i]]
            #print(f"feature {feature_name} i: {feature_importances_indices[i]}")

            #match_for_categorical_spline_or_interaction =  any([re.match("C\(.*\)\[.*\]", feature_name), feature_name.startswith("bs("), ":" in feature_name])
            #if match_for_categorical_spline_or_interaction:
                #print("skipping")
            #    continue
            #print(f"top1000_individuals_indexes: {top1000_individuals_indexes}")
            #print(f"testX[top1000_individuals_indexes, feature_importances_indices[i]].shape: {testX[top1000_individuals_indexes, feature_importances_indices[i]].shape}")


            top20percentiles_boxplot_stats = get_boxplot_stats(testX[top20percentile_individuals_indexes, feature_importances_indices[i]])
            bottom20percentiles_boxplot_stats = get_boxplot_stats(testX[bottom20percentiles_individuals_indexes, feature_importances_indices[i]])

            model_perf_summary["test"]["important_feature_boxplots__top20percentiles"].append((i, feature_name, top20percentiles_boxplot_stats))
            model_perf_summary["test"]["important_feature_boxplots__bottom20percentiles"].append((i, feature_name, bottom20percentiles_boxplot_stats))
            
        positives_individuals_indexes = np.argwhere(testY == 1)
        negatives_individuals_indexes = np.argwhere(testY == 0)
        model_perf_summary["test"]["important_feature_boxplots__positives"] = []
        model_perf_summary["test"]["important_feature_boxplots__negatives"] = []
        for i in range(min(10, len(feature_importances_indices))):
            feature_name = patient_feature_vector___index__to__feature__list[feature_importances_indices[i]]
            #print(f"feature {feature_name} i: {feature_importances_indices[i]}")

            #match_for_categorical_spline_or_interaction =  any([re.match("C\(.*\)\[.*\]", feature_name), feature_name.startswith("bs("), ":" in feature_name])
            #if match_for_categorical_spline_or_interaction:
                #print("skipping")
            #    continue
            #print(f"top1000_individuals_indexes: {top1000_individuals_indexes}")
            #print(f"testX[top1000_individuals_indexes, feature_importances_indices[i]].shape: {testX[top1000_individuals_indexes, feature_importances_indices[i]].shape}")


            positives_boxplot_stats = get_boxplot_stats(testX[positives_individuals_indexes, feature_importances_indices[i]])
            negatives_boxplot_stats = get_boxplot_stats(testX[negatives_individuals_indexes, feature_importances_indices[i]])

            model_perf_summary["test"]["important_feature_boxplots__positives"].append((i, feature_name, positives_boxplot_stats))
            model_perf_summary["test"]["important_feature_boxplots__negatives"].append((i, feature_name, negatives_boxplot_stats))


    if do_plots:
        test_confusion_matrix = model_perf_summary["test"]["confusion_matrix"]
        plt.figure(num=None, figsize=(9,6), dpi=96)
        plt.subplot(3 if do_feature_importances else 3,3,(1,4))
        plt.imshow(test_confusion_matrix, cmap=plt.cm.coolwarm, interpolation="nearest")

        for x in range(test_confusion_matrix.shape[0]):
            for y in range(test_confusion_matrix.shape[1]):
                plt.annotate(str(test_confusion_matrix[x][y]), xy=(y,x), horizontalalignment="center", verticalalignment="center")
        plt.yticks([0, 1], ["Control", "Case"])
        plt.xticks([0, 1], ["Control", "Case"])
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title("Test Confusion Matrix")

        fpr, tpr, thresholds = roc_curve(y_true = testY, y_score=model.predict_proba(testX)[:,1], pos_label=1)
        plt.subplot(3,3,(2,5))
        plt.plot(fpr, tpr, color="darkorange", lw=1, label="ROC curve")
        plt.plot([0,1],[0,1],color='navy', lw=1, linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test ROC Curve")
        #plt.legend(loc="lower right")

        precision, recall, thresholds = precision_recall_curve(y_true = testY, probas_pred=model.predict_proba(testX)[:,1], pos_label=1)
        plt.subplot(3,3,(3,6))
        plt.plot(recall, precision, color="darkorange", lw=1, label="Precision-Recall curve")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Test Precision-Recall Curve")
        #plt.legend(loc="lower right")

        if do_feature_importances:
            num_features = min(10, len(feature_importances))
            plt.subplot(3,3,(7,9))
            plt.bar(list(range(num_features)), feature_importances[feature_importances_indices[:num_features]], yerr=feature_importances_std[feature_importances_indices[:num_features]], align="center")
            plt.xticks(list(range(num_features)), [patient_feature_vector___index__to__feature__list[i] for i in feature_importances_indices[:num_features]], rotation=45)
            plt.title("Feature importances")
            plt.ylabel("Importance Ratio")
            plt.xlabel("")

        plt.tight_layout()
        #plt.show() #|| saveplot performance

#%%
    ppv_dict_size = 0
    if do_plots:
        ppv_at_x_dict = model_perf_summary["test"]["ppv_at_X"]
        ppv_dict_size = len(ppv_at_x_dict)

    if do_plots and ppv_dict_size > 10:

#        x1 = np.linspace(1, ppv_dict_size, ppv_dict_size/5)
#        x2 = np.linspace(1, 500, 500/2)
#        ppv_interpolated = interp1d(list(ppv_at_x_dict.keys()), [ppv_at_X["ppv"] for ppv_at_X in ppv_at_x_dict.values()], kind="cubic")
#        drr_interpolated = interp1d(list(ppv_at_x_dict.keys()), [ppv_at_X["drr"] for ppv_at_X in ppv_at_x_dict.values()], kind="cubic")
#
#
#        mpl.style.use("default")
#        plt.figure(num=None, figsize=(9,5), dpi=96)
#
#        ax1 = plt.subplot(2,1,1)
#        ax1.set_title("PPV for top predicted individuals")
#        ax1.plot(x1, ppv_interpolated(x1), 'C1')
#        plt.ylabel("PPV", color="C1")
#        ax2 = ax1.twinx()
#        ax2.plot(x1, drr_interpolated(x1), 'C2')
#        plt.ylabel("Death relative risk", color="C2")
#
#        ax3 = plt.subplot(2,1,2)
#        ax3.plot(x2, ppv_interpolated(x2), 'C1')
#        plt.ylabel("PPV", color="C1")
#        ax4 = ax3.twinx()
#        ax4.plot(x2, drr_interpolated(x2), 'C2')
#        plt.ylabel("Death relative risk", color="C2")

        #plt.show() #|| saveplot ppv_at_X

        mpl.style.use("default")
        plt.figure(num=None, figsize=(9,5), dpi=96)
        ax1 = plt.subplot(2,1,1)
        ax1.set_title("PPV for top predicted individuals")
        ax1.plot(list(ppv_at_x_dict.keys()), [ppv_at_X["ppv"] for ppv_at_X in ppv_at_x_dict.values()], 'C1')
        plt.ylabel("PPV", color="C1")
        ax2 = ax1.twinx()
        ax2.plot(list(ppv_at_x_dict.keys()), [ppv_at_X["drr"] for ppv_at_X in ppv_at_x_dict.values()], 'C2')
        plt.ylabel("Death relative risk", color="C2")

        ax3 = plt.subplot(2,1,2)
        ax3.plot(list(ppv_at_x_dict.keys())[:100], list([ppv_at_X["ppv"] for ppv_at_X in ppv_at_x_dict.values()])[:100], 'C1')
        plt.ylabel("PPV", color="C1")
        ax4 = ax3.twinx()
        ax4.plot(list(ppv_at_x_dict.keys())[:100], list([ppv_at_X["drr"] for ppv_at_X in ppv_at_x_dict.values()])[:100], 'C2')
        plt.ylabel("Death relative risk", color="C2")
        #plt.show() #|| saveplot ppv_at_X__no_interpolation
#%%



#        plt.figure(num=None, figsize=(9,5), dpi=96)
#        plt.title("PPV for top predicted individuals")
#        ppv_at_x_dict = model_perf_summary["test"]["ppv_at_X"]
#        plt.subplot(2,1,1)
#        plt.title("PPV for top predicted individuals")
#        plt.plot(list(ppv_at_x_dict.keys()), [ppv_at_X["ppv"] for ppv_at_X in ppv_at_x_dict.values()], '-')
#        plt.ylabel("PPV")
#        ax2 = plt.twinx()
#        ax2.plot(list(ppv_at_x_dict.keys()), [ppv_at_X["drr"] for ppv_at_X in ppv_at_x_dict.values()], '-', 'r')
#        ax2.ylabel("Death relative risk")
#
#        plt.subplot(2,1,2)
#        plt.plot(list(ppv_at_x_dict.keys())[:500], list([ppv_at_X["ppv"] for ppv_at_X in ppv_at_x_dict.values()])[:500], '-')
#
#        plt.xlabel("# top predicted individuals")
#        plt.ylabel("PPV")
#        plt.show() #|| saveplot ppv_at_X


    if do_plots and do_feature_importances and len(feature_importances_indices) > 10: #The below doesn't make sense for the baseline model
        n_features = 50
        features = [patient_feature_vector___index__to__feature__list[feature_importances_indices[i]] for i in range(len(patient_feature_vector___index__to__feature__list))]
        #y= [feature_importances[feature_importances_indices[i]] for i in range(n_features)]

        categories = list(map(lambda i: i if i in {"DIAG", "ATC", "SKS_OPR", "SSR_SPECIALE", "PDIAG"} else "AUX", [i.split("-")[0] for i in features]))
        category_map, categories_i = np.unique(categories, return_inverse=True)

        #category_plot_handles = []
        legend_desc_map = {
                "DIAG": "Diagnoses",
                "PDIAG": "Parents diagnoses",
                "ATC": "Prescriptions",
                "SKS_OPR": "Hospital procedures",
                "SSR_SPECIALE": "Non-hospital health services",
                "AUX": "Date of birth, date of first t2d diag, gender, country of birth"
                }

        plt.figure(num=None, figsize=(14.6,4.9), dpi=96)
        for cat_i, cat_name in enumerate(category_map):
            x=[i for i in range(n_features) if categories_i[i] == cat_i]
            y=[feature_importances[feature_importances_indices[i]] for i in range(n_features) if categories_i[i] == cat_i]
            cumulative_importance = sum([feature_importances[feature_importances_indices[i]] for i in range(len(feature_importances)) if categories_i[i] == cat_i])
            if len(x):
                plt.bar(x, y, label='{}{}(cumulative importance: {:.2f})'.format(legend_desc_map[cat_name], "\n" if len(legend_desc_map[cat_name])>30 else " ", float(cumulative_importance)))

        plt.xticks([])
        plt.xlim([-1, n_features + 0.5])
        plt.ylabel("Feature importance")
        plt.xlabel("Features")
        plt.legend().get_frame().set_alpha(0.25)
        plt.show() #|| saveplot combined_feature_importance
#        plt.text(22, 0.05,  #It's relative to data! not 0,0 of the plot
#                 f"Cumulative importance of first {n_features} features: {sum([feature_importances[feature_importances_indices[i]] for i in range(n_features)]):.2f} \n" +\
#                 f"Cumulative importance of all the other features: {sum([feature_importances[feature_importances_indices[i]] for i in range(n_features, len(feature_importances))]):.2f}",
#                 verticalalignment="top",
#                 bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.25},
#                 )
        print(f"Cumulative importance of first {n_features} features: {sum([feature_importances[feature_importances_indices[i]] for i in range(n_features)]):.2f}")
        print(f"Cumulative importance of all the other features: {sum([feature_importances[feature_importances_indices[i]] for i in range(n_features, len(feature_importances))]):.2f}")


def plot_prformance_curves_for_metric(fit_results, metric_name):
    available_params = [p_name for p_name, val in fit_results[0]['params']]

    metric_param_performance = []
    for fit_result in fit_results:
        metric_param_performance.append((fit_result['avg_scores'][metric_name], {p_name: val for p_name, val in fit_result['params']}))
    best_metric_params = sorted(metric_param_performance, key=lambda p: p[0], reverse=True)[0][1]

    plt.figure()
    plt.title("Parameter performance curves for {} metric".format(metric_name))
    for pi, free_param in enumerate(available_params):
        metric_vals = []
        free_param_vals = []
        fixed_params = {p_name: val for p_name, val in list(best_metric_params.items()) if p_name != free_param}
        #print(fixed_params)
        for metric_val, params in metric_param_performance:
            if set(fixed_params.items()) == set([(p_name, p_val) for p_name, p_val in list(params.items()) if p_name != free_param]):
                #print(p_val)
                metric_vals.append(metric_val)
                free_param_vals.append(params[free_param])
        #print(free_param)
        #print(type(free_param_vals[0]))
        plt.subplot(len(available_params), 1, pi+1)
        if type(free_param_vals[0]) == str:
            free_param_vals_map = {val: i for i, val in enumerate(set(free_param_vals))}
            plt.bar([free_param_vals_map[val] for val in free_param_vals], metric_vals, tick_label=[val for val, i in sorted(iter(list(free_param_vals_map.items())), key=lambda p:p[1])])
        else:
            plt.plot(free_param_vals, metric_vals)
        plt.xlabel(free_param)
        plt.ylabel(metric_name)
    plt.tight_layout()
    plt.show() #|| saveplot model_parameter_performance

def get_best_models_fit_results_based_on_training_data(fit_results):
    """
    Return fit results for best model for each score metric
    """

    def dict_from_tuple(t):
        return {param_name: param_value for param_name, param_value in t}

    best_models = {}
    for score in fit_results[0]["avg_scores"]:

        if score == "ROC AUC" or score == "Precision-Recall Curve AUC":
            sig_score_diff = 0.005
        elif score.startswith("ppv") or score == "precision" or score == "recall" or score.startswith("specificity") or score.startswith("sensitivity"):
            sig_score_diff = 0.005
        elif score.startswith("F-") or score.startswith("Accuracy"):
            sig_score_diff = 0.005
        elif score == "ROC AUC CI":
            continue
        else:
            sig_score_diff = 0.00001

        model_type_name = str(type(fit_results[0]["combined_model"]))

        if "LogisticRegression" in model_type_name:
            complexity_parameter = "C"
        elif "XGBClassifier" in model_type_name:
            complexity_parameter = "max_depth"
        elif "RandomForestClassifier" in model_type_name:
            complexity_parameter = "max_depth"
        #elif "LogisticRegression" in str(type(fit_results["combined_model"])):
        #    complexity_parameter = "C"
        else:
            raise f"Couldn't handle type {model_type_name}"


        sorted_fit_results = sorted(fit_results, key=lambda result: result["avg_scores"][score] if score in result["avg_scores"] else 0, reverse=True)

        #get all results matching the sig_score_diff criterion
        equivalent_results = [result for result in sorted_fit_results if (sorted_fit_results[0]["avg_scores"][score] - result["avg_scores"][score]) < sig_score_diff]
        #print(sorted_fit_results[0]['params'])
        #now find the model with smallest complexity_parametere within the equivalent_results
        best_models[score] = sorted(equivalent_results, key=lambda result: dict_from_tuple(result["params"])[complexity_parameter] if score in result["avg_scores"] else 0, reverse=False)[0]


        #best_models[score] = sorted(fit_results, key=lambda result: result["avg_scores"][score] if score in result["avg_scores"] else 0, reverse=True)[0]

    return best_models

def print_best_model_summary_based_on_training_data(best_models):
    print("\n\n#\n#Best model scores for each metric (training):\n#")
    for score in best_models:
        print(("{:28s}  score:{}\t params: {}".format(score, best_models[score]["avg_scores"][score], ", ".join(["{}: {}".format(s,v) for s, v in best_models[score]['params']]))))

    print("\n\n#\n#Best model in F-2\n#")
    print("scores (training):")
    for score, value in list(best_models['F-2']["avg_scores"].items()):
        pprint("{:28s}: {}".format(score ,value))
    print("params:")
    for param, value in best_models['F-2']["params"]:
        pprint("{}: {}".format(param ,value))

    print("\n\n#\n#Best model in ROC AUC\n#")
    print("scores (training):")
    pprint(best_models['ROC AUC']["avg_scores"])
    print("params:")
    pprint(best_models['ROC AUC']["params"])
