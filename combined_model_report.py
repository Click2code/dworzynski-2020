# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:25:35 2017

@author: fskPioDwo
"""

from itertools import product
from collections import Counter, defaultdict
from pprint import pprint, pformat
import time
from glob import glob
import pickle
from os import path, mkdir
import json
from zipfile import ZipFile, ZIP_DEFLATED
from common import ready_for_json

import numpy as np

from model_evaluation import *

#%%

class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()

        # Let the base class default method raise the TypeError
        return super(CustomJsonEncoder, self).default(o)
#%%
def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    else:
        str(val)

def generate_and_save_combined_model_report(pipeline_name, pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"):
    print(f"Generating report for {pipeline_name}")
    pipeline_dir = path.join(pipelines_dir, pipeline_name)
    pipeline_data_dir = path.join(pipeline_dir, "data")
    model_fit_results_fpaths = glob(pipeline_data_dir + "\\*model_type_summary.pickle")
    model_fit_results = {}
    for p in model_fit_results_fpaths:
        model_run_name = path.split(p)[1].split("__model_type_summary.pickle")[0]
        model_fit_results[model_run_name] = pickle.load(open(p,"rb"), encoding='latin1')
        
    if path.exists(pipeline_data_dir + "\\method_auroc_diff_CIs.pickle"):
        model_fit_results["method_auroc_diff_CIs"] = pickle.load(open(pipeline_data_dir + "\\method_auroc_diff_CIs.pickle","rb"), encoding='latin1')
    if path.exists(pipeline_data_dir + "\\uncalibrated_model_expected_calibration_error.pickle"):
        model_fit_results["uncalibrated_model_expected_calibration_error"] = pickle.load(open(pipeline_data_dir + "\\uncalibrated_model_expected_calibration_error.pickle","rb"), encoding='latin1')
    if path.exists(pipeline_data_dir + "\\calibrated_model_expected_calibration_error.pickle"):
        model_fit_results["calibrated_model_expected_calibration_error"] = pickle.load(open(pipeline_data_dir + "\\calibrated_model_expected_calibration_error.pickle","rb"), encoding='latin1')
    
    #See how it looks using:
    #print(pformat(next(iter(model_fit_results.items()))))
    #{'LR_aux__': {'F-2 combined': {'test': {'classification_report': '             precision    recall  f1-score   support\n\n   controls       0.88      0.60      0.71     15916\n      cases       0.24      0.61      0.34      3277\n\navg / total       0.77      0.60      0.65     19193\n',
    #    'confusion_matrix': array([[9571, 6345],
    #           [1291, 1986]]),
    #    'metrics': {'Accuracy': 0.60214661595373309,
    #     'F-1': 0.34217780840799444,
    #     'F-2': 0.46317458836699471,
    #     'Precision-Recall Curve AUC': 0.25011963570992923,
    #     'ROC AUC': 0.64287605289380478,
    #     'precision': 0.23838674828952106,
    #     'recall': 0.60604211168751909}}, ...
    # then ...
    # 'method_auroc_diff_CIs': {'gb_bs_auroc_diff_CI': {'lower': 0.053010989804580855,
    #                     'mean': 0.067646280155515257,
    #                     'median': 0.06773865210193597,
    #                     'upper': 0.08342030522517474},
    # 'gb_lr_auroc_diff_CI': {'lower': 0.022039442497104433,
    #                     'mean': 0.032375835443391551,
    #                     'median': 0.032429038104142882,
    #                     'upper': 0.042562159303281444},
    # .....
    summary_stat_values_dir = path.join(pipeline_dir, "values")
    summary_stat_values_fpaths = glob(summary_stat_values_dir + "\\*.pickle")
    summary_stat_values = {path.split(p)[1].split(".pickle")[0]: pickle.load(open(p,"rb"), encoding='latin1') for p in summary_stat_values_fpaths}

    print(f"Dumping results_summary for {pipeline_name}")
    try:
        json.dump(ready_for_json(model_fit_results), open(path.join(pipeline_dir, "results_summary.json"), "w"))
    except Exception as e:
        print("PROBLEM: "+ path.join(pipeline_dir, "results_summary.json"))
        raise e

    print(f"Dumping stat_values_combined for {pipeline_name}")
    try:
        json.dump(ready_for_json(summary_stat_values), open(path.join(pipeline_dir, "stat_values_combined.json"), "w"))
    except Exception as e:
        print(path.join(pipeline_dir, "stat_values_combined.json"))
        raise e



def merge_reports(pipeline_names, merged_report_name, pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"):
    combined_report = {}
    #Save combined model fit report
    for pipeline_name in pipeline_names:
        combined_report[pipeline_name] = json.load(open(path.join(pipelines_dir, pipeline_name, 'results_summary.json'), "r"))


    combined_stat_values_report = {}
    #Save combined model values report
    for pipeline_name in pipeline_names:
        combined_stat_values_report[pipeline_name] = json.load(open(path.join(pipelines_dir, pipeline_name, 'stat_values_combined.json'), "r"))

    #Save plots in a combined archive
#    with ZipFile(path.join(pipelines_dir, merged_report_name + ".zip"), "w", compression=ZIP_DEFLATED) as zipf:
#        for pipeline_name in pipeline_names:
#            pipeline_plot_dir = path.join(pipelines_dir, pipeline_name, "plots")
#            pipeline_plots_fpaths = glob(pipeline_plot_dir + "\\*.png")
#            for plot_file_path in pipeline_plots_fpaths:
#                zipf.write(plot_file_path, merged_report_name + "\\" + pipeline_name + "\\" + path.split(plot_file_path)[1])

    #Create plot directory for whole report if doesn't exits
    report_plots_dir_path = path.join(pipelines_dir, merged_report_name)
    if not path.exists(report_plots_dir_path):
        mkdir(report_plots_dir_path)


    for pipeline_name in pipeline_names:
        pipeline_plot_dir = path.join(pipelines_dir, pipeline_name, "plots")
        pipeline_plots_fpaths = glob(pipeline_plot_dir + "\\*.png")
        with ZipFile(path.join(report_plots_dir_path, pipeline_name + ".zip"), "w", compression=ZIP_DEFLATED) as zipf:
            for pipeline_plot_file_path in pipeline_plots_fpaths:
                zipf.write(pipeline_plot_file_path, pipeline_name + "\\" + path.split(pipeline_plot_file_path)[1])

    json.dump(ready_for_json(combined_report), open(path.join(pipelines_dir, merged_report_name + ".json"), "w"), default=to_serializable)
    json.dump(ready_for_json(combined_stat_values_report), open(path.join(pipelines_dir, merged_report_name + "_stat_values.json"), "w"))
