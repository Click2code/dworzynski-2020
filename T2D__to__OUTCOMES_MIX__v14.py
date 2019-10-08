# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:18:34 2017

@author: fskPioDwo
"""

from pipeline_1_1 import Action, Pipeline, PipelineRun
import networkx as nx
import pickle
from collections import defaultdict
import numpy as np
from datetime import datetime
from time import sleep

def code_range_generator(first_code_letter, code_digits_range_start, code_digits_range_end):
    code_digits_range = range(int(code_digits_range_start), int(code_digits_range_end)+1)
    codes = ["{}{:02d}".format(first_code_letter, code_digits) for code_digits in code_digits_range]
    return codes


execution_plan = [

        ("Load LPR raw data", "Load LPR raw data", "", {}),
        ("Load CPR data", "Load CPR data", "", {}),
        ("Load Birth Register data and identify women and their pregnancy periods",  "Load Birth Register data and identify women and their pregnancy periods", "", {}),
        ("Identify subpopulation individuals from LPR", "Identify subpopulation individuals from LPR", "", {}),
        ("Load and identify subpopulation individuals from LMS", "Load and identify subpopulation individuals from LMS","" , {}),
        #("Remove females below 40 yo", "Remove females below 40 yo", "" , {}),
        ("Merge subpopulation LBNRs and first event dataframes identified in LMS and LPR", "Merge subpopulation LBNRs and first event dataframes identified in MRF and LPR", "" , {}),
        ("Extract subpopulation from LPR", "Extract subpopulation from LPR", "" , {}),
        ("Extract subpopulation from LMS", "Extract subpopulation from LMS", "" , {}),
        ("Extract subpopulation from CPR", "Extract subpopulation from CPR", "" , {}),
        ("Load Death register data and extract subpopulation" , "Load Death register data and extract subpopulation", "", {}),
        ("Extract subpopulation parents from LPR", "Extract subpopulation parents from LPR", "" , {}),
        ("Extract subpopulation from Address data", "Extract subpopulation from Address data", "" , {}),
        ("Extract subpopulation from LPR Procedures", "Extract subpopulation from LPR Procedures","" , {}),
        ("Extract subpopulation from SSR", "Extract subpopulation from SSR", "" , {}),

        ("Filter events and individuals", "Filter events and individuals", "", {}),
        ("Divide feature set into training, test and validation. Generate K-fold indexes.", "Divide feature set into training, test and validation. Generate K-fold indexes.", "", {}),



    ("Run Logistic Regression", "Run Logistic Regression on age timeoft2d sex", "LR_aux", {
        "trainX": "aux__trainX"}),
    ("Describe Fit Results", "Describe Fit Results of Logistic Regression on on age timeoft2d sex", "LR_aux", {
        "trainX": "aux__trainX",
        "testX": "aux__testX",
        "valX": "aux__valX",
        "patient_feature_vector___index__to__feature__list": "aux__patient_feature_vector___index__to__feature__list",
        "fit_results": "LR_aux__fit_results"}),

    ("Run Random Forest", "Run Random Forest on count data", "RandFor_count", {
        "trainX": "count__trainX"}),
    ("Describe Fit Results", "Describe Fit Results of Random Forest on count data", "RandFor_count", {
        "trainX": "count__trainX",
        "testX": "count__testX",
        "valX": "count__valX",
        "fit_results": "RandFor_count__fit_results"}),

    ("Run Logistic Regression", "Run Logistic Regression on count data", "LR_count", {
        "trainX": "count__trainX"}),
    ("Describe Fit Results", "Describe Fit Results of Logistic Regression on count data", "LR_count", {
        "trainX": "count__trainX",
        "testX": "count__testX",
        "valX": "count__valX",
        "fit_results": "LR_count__fit_results"}),



    ("Run Extreme Gradient Boosting", "Run Extreme Gradient Boosting on count data", "ExGradientBoost_count", {
        "trainX": "count__trainX"}),
    ("Describe Fit Results", "Describe Fit Results of Extreme Gradient Boosting on balanced count data", "ExGradientBoost_count", {
        "trainX": "count__trainX",
        "testX": "count__testX",
        "valX": "count__valX",
        "fit_results": "ExGradientBoost_count__fit_results"}),

    ("Calculate method AUROC diffs", "Calculate method AUROC diffs", "" , {
            "aux_testX": "aux__testX",
            "aux_valX": "aux__valX",
            "count_testX": "count__testX",
            "count_valX": "count__valX",
            "bs_fit_results": "LR_aux__fit_results",
            "lr_fit_results": "LR_count__fit_results",
            "rf_fit_results": "RandFor_count__fit_results",
            "gb_fit_results": "ExGradientBoost_count__fit_results",
            }),
    
    
    ("Calibrate models and describe their fit results", "Calibrate models, save calibration curves pre and post calibration, generate model_summary for calibrated models", "", {
        "aux_trainX": "aux__trainX",
        "aux_testX": "aux__testX",
        "aux_valX": "aux__valX",
        "count_trainX": "count__trainX",
        "count_testX": "count__testX",
        "count_valX": "count__valX",    
        "blr_fit_results": "LR_aux__fit_results",
        "lr_fit_results": "LR_count__fit_results",
        "rf_fit_results": "RandFor_count__fit_results",
        "xgb_fit_results": "ExGradientBoost_count__fit_results"
        })

]


param_values = {
    "raw_data_dir_path": "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\",
    "diagnoses_t_diag_types": {"A", "B", "G", "C", "+"},
    "subpopulation_icd_code_pattern": "E11",
    "subpopulation_ATC_code_pattern": "A10B", #A10 is "drugs used in diabetes"- including type 1 diabetes
    "age_restricted_subpopulation_ATC_code_sub_pattern": "A10A",
    "subpopulation_min_age_at_first_restricted_presc_event": 30,
    "subpopulation_days_offset": 0,
    "pre_t0_time_window": -200, #It's 200 years, so basically no window is applied
    "post_t0_time_window": 0,
    "last_calendar_day_of_data": datetime(2016, 1, 1),
    "earliest_required_year_of_diagnosis": 2000, #chosen based on histogram of incidence
    "followup_period_years": 5,
    #"year_to_count_as_zero": 1850, #dates are counted from this moment in months
    #"min_female_age": 40,
    "drop_subpop_events_during_pregnancies": True,
    "use_death_registry": True,
    "buffer_period_length": 30,

    "best_model_criterion": "ROC AUC",

    #This is including parents data - not found to be highly predictive, low coverage and shoves additional 1200 features in
    "use_parent_diagnoses": False,
    "subpop_parents_diag_df": None,

    "diag_feature_min_support": 50,
    "sksopr_feature_min_support": 50,
    "prescription_feature_min_support": 50,
    "outpatient_visit_feature_min_support": 50,

    "num_k_folds": 3,
    "use_time_based_split": False,


    "use_diag_codes_up_to_length": 4,
    "use_proc_codes_up_to_length": 5,
    "use_drug_codes_up_to_length": 5,


    "training_set_ratio": 0.7,
    "test_set_ratio": 0.2,

    "n_PCA_components": 100,

    "rf_param_grid": {
         "class_weight": ["balanced_subsample"],
         "n_estimators": [1000, 1500],
         "max_depth": [12, 14, 16],
         "max_features": ["sqrt"],
         "criterion": ["gini"]
    },

    "logireg_param_grid": {
        #"penalty": ["l2"],
        "class_weight": ["balanced"], #class weights inversely proportional to frequency
        "C": [0.6, 0.7, 0.8],
        "max_iter": [300, 400, 500],
    },

    "svd_param_grid": {
        "C": [0.9, 0.8, 0.7, 0.6]
    },

    "nn_param_grid": {
        "hidden_layer_sizes": [(100, 100)],
       "alpha": [0.0001],
       "batch_size": [1000]
    },

    "xgb_param_grid": {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.025, 0.05, 0.1],
        "n_estimators": [200], #this is almost always overriden by early stopping
    }
}

#%%
    
#Test run without class weights
    

#sleep(60*60)
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit_no_weights__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()

    
    

#%%
#%%
#sleep(60*60)
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["buffer_period_length"] = 60

r = PipelineRun("T2D_timesplit__CKD_N17-N19_60days__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
#I20 - angina

param_values["outcome_diagnosis_pattern"] = ("I")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit__CVD_all__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#I20 - angina

param_values["outcome_diagnosis_pattern"] = ("I20", "I24", "I25", "I679")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit__CVD_no_HF_MI_ST__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
#sleep(60*60)
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#sleep(2*60*60)
param_values["outcome_diagnosis_pattern"] = ("I50")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit__CVD_HF_I50__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I21", "I22", "I23")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit__MI__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I61", "I62", "I63", "I64")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit__Stroke__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ()
param_values["use_death_as_outcome"] = True
param_values["use_time_based_split"] = True

r = PipelineRun("T2D_timesplit__death__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

#
# No timesplit
#


#%%
#%%
execution_plan = [

        ("Load LPR raw data", "Load LPR raw data", "", {}),
        ("Load CPR data", "Load CPR data", "", {}),
        ("Load Birth Register data and identify women and their pregnancy periods",  "Load Birth Register data and identify women and their pregnancy periods", "", {}),
        ("Identify subpopulation individuals from LPR", "Identify subpopulation individuals from LPR", "", {}),
        ("Load and identify subpopulation individuals from LMS", "Load and identify subpopulation individuals from LMS","" , {}),
        #("Remove females below 40 yo", "Remove females below 40 yo", "" , {}),
        ("Merge subpopulation LBNRs and first event dataframes identified in LMS and LPR", "Merge subpopulation LBNRs and first event dataframes identified in MRF and LPR", "" , {}),
        ("Extract subpopulation from LPR", "Extract subpopulation from LPR", "" , {}),
        ("Extract subpopulation from LMS", "Extract subpopulation from LMS", "" , {}),
        ("Extract subpopulation from CPR", "Extract subpopulation from CPR", "" , {}),
        ("Load Death register data and extract subpopulation" , "Load Death register data and extract subpopulation", "", {}),
        ("Extract subpopulation parents from LPR", "Extract subpopulation parents from LPR", "" , {}),
        ("Extract subpopulation from Address data", "Extract subpopulation from Address data", "" , {}),
        ("Extract subpopulation from LPR Procedures", "Extract subpopulation from LPR Procedures","" , {}),
        ("Extract subpopulation from SSR", "Extract subpopulation from SSR", "" , {}),

        ("Filter events and individuals", "Filter events and individuals", "", {}),
        ("Divide feature set into training, test and validation. Generate K-fold indexes.", "Divide feature set into training, test and validation. Generate K-fold indexes.", "", {}),




    ("Run Extreme Gradient Boosting", "Run Extreme Gradient Boosting on count data", "ExGradientBoost_count", {
        "trainX": "count__trainX"}),
    ("Describe Fit Results", "Describe Fit Results of Extreme Gradient Boosting on balanced count data", "ExGradientBoost_count", {
        "trainX": "count__trainX",
        "testX": "count__testX",
        "valX": "count__valX",
        "fit_results": "ExGradientBoost_count__fit_results"}),


]
#%%
param_values["outcome_diagnosis_pattern"] = ("I")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__CVD_all__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I20", "I24", "I25", "I679")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__CVD_no_HF_MI_ST__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I50")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__CVD_HF_I50__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I21", "I22", "I23")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__MI__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I61", "I62", "I63", "I64")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__Stroke__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ()
param_values["use_death_as_outcome"] = True
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__death__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#done
param_values["outcome_diagnosis_pattern"] = ("E113")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__OpthalmicComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("E114")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__NeuroComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#####
# These are extras for comparison purpose
#####
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = False

r = PipelineRun("T2D_notimesplit__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%#####
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["best_model_criterion"] = "Precision-Recall Curve AUC"

r = PipelineRun("T2D_timesplit_auPRC__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

#####
# T+1
#####

execution_plan = [

        ("Load LPR raw data", "Load LPR raw data", "", {}),
        ("Load CPR data", "Load CPR data", "", {}),
        ("Load Birth Register data and identify women and their pregnancy periods",  "Load Birth Register data and identify women and their pregnancy periods", "", {}),
        ("Identify subpopulation individuals from LPR", "Identify subpopulation individuals from LPR", "", {}),
        ("Load and identify subpopulation individuals from LMS", "Load and identify subpopulation individuals from LMS","" , {}),
        #("Remove females below 40 yo", "Remove females below 40 yo", "" , {}),
        ("Merge subpopulation LBNRs and first event dataframes identified in LMS and LPR", "Merge subpopulation LBNRs and first event dataframes identified in MRF and LPR", "" , {}),
        ("Extract subpopulation from LPR", "Extract subpopulation from LPR", "" , {}),
        ("Extract subpopulation from LMS", "Extract subpopulation from LMS", "" , {}),
        ("Extract subpopulation from CPR", "Extract subpopulation from CPR", "" , {}),
        ("Load Death register data and extract subpopulation" , "Load Death register data and extract subpopulation", "", {}),
        ("Extract subpopulation parents from LPR", "Extract subpopulation parents from LPR", "" , {}),
        ("Extract subpopulation from Address data", "Extract subpopulation from Address data", "" , {}),
        ("Extract subpopulation from LPR Procedures", "Extract subpopulation from LPR Procedures","" , {}),
        ("Extract subpopulation from SSR", "Extract subpopulation from SSR", "" , {}),

        ("Filter events and individuals", "Filter events and individuals", "", {}),
        ("Divide feature set into training, test and validation. Generate K-fold indexes.", "Divide feature set into training, test and validation. Generate K-fold indexes.", "", {}),

    ("Run Logistic Regression", "Run Logistic Regression on age timeoft2d sex", "LR_aux", {
        "trainX": "aux__trainX"}),
    ("Describe Fit Results", "Describe Fit Results of Logistic Regression on on age timeoft2d sex", "LR_aux", {
        "trainX": "aux__trainX",
        "testX": "aux__testX",
        "valX": "aux__valX",
        "patient_feature_vector___index__to__feature__list": "aux__patient_feature_vector___index__to__feature__list",
        "fit_results": "LR_aux__fit_results"}),

    ("Run Extreme Gradient Boosting", "Run Extreme Gradient Boosting on count data", "ExGradientBoost_count", {
        "trainX": "count__trainX"}),
    ("Describe Fit Results", "Describe Fit Results of Extreme Gradient Boosting on balanced count data", "ExGradientBoost_count", {
        "trainX": "count__trainX",
        "testX": "count__testX",
        "valX": "count__valX",
        "fit_results": "ExGradientBoost_count__fit_results"}),

]

#%% Done
param_values["outcome_diagnosis_pattern"] = ("I20", "I24", "I25", "I679")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__CVD_no_HF_MI_ST__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%

#%% Done
param_values["outcome_diagnosis_pattern"] = ("I")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__CVD_all__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I50")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__CVD_HF_I50__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I21", "I22", "I23")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__MI__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I61", "I62", "I63", "I64")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__Stroke__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ()
param_values["use_death_as_outcome"] = True
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__death__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#done
param_values["outcome_diagnosis_pattern"] = ("E113")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__OpthalmicComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("E114")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*1

r = PipelineRun("T2D_1_timesplit__NeuroComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_1__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

#####
# T+2
#####

#%% Done
param_values["outcome_diagnosis_pattern"] = ("I20", "I24", "I25", "I679")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__CVD_no_HF_MI_ST__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%% Done
sleep(60*60)
param_values["outcome_diagnosis_pattern"] = ("I")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__CVD_all__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I50")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__CVD_HF_I50__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I21", "I22", "I23")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__MI__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I61", "I62", "I63", "I64")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__Stroke__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ()
param_values["use_death_as_outcome"] = True
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__death__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("E113")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__OpthalmicComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("E114")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*2

r = PipelineRun("T2D_2_timesplit__NeuroComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_2__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

#####
# T+3
#####

#%%
param_values["outcome_diagnosis_pattern"] = ("I")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__CVD_all__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I20", "I24", "I25", "I679")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__CVD_no_HF_MI_ST__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I50")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__CVD_HF_I50__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I21", "I22", "I23")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__MI__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I61", "I62", "I63", "I64")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__Stroke__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ()
param_values["use_death_as_outcome"] = True
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__death__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#done
param_values["outcome_diagnosis_pattern"] = ("E113")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__OpthalmicComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("E114")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*3

r = PipelineRun("T2D_3_timesplit__NeuroComplic__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_3__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()


#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

#####
# T+4
#####
#%%
param_values["outcome_diagnosis_pattern"] = ("I")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*4

r = PipelineRun("T2D_4_timesplit__CVD_all__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_4__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I20", "I24", "I25", "I679")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*4

r = PipelineRun("T2D_4_timesplit__CVD_no_HF_MI_ST__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_4__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = tuple(code_range_generator("N", 17, 19))
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*4

r = PipelineRun("T2D_4_timesplit__CKD_N17-N19__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_4__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I50")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*4

r = PipelineRun("T2D_4_timesplit__CVD_HF_I50__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_4__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
param_values["outcome_diagnosis_pattern"] = ("I21", "I22", "I23")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*4

r = PipelineRun("T2D_4_timesplit__MI__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_4__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("I61", "I62", "I63", "I64")
param_values["use_death_as_outcome"] = False
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*4

r = PipelineRun("T2D_4_timesplit__Stroke__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_4__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ()
param_values["use_death_as_outcome"] = True
param_values["use_time_based_split"] = True
param_values["subpopulation_days_offset"] = 365*4

r = PipelineRun("T2D_4_timesplit__death__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D_4__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%


#%%
param_values["outcome_diagnosis_pattern"] = ("C91")
param_values["use_death_as_outcome"] = False

r = PipelineRun("T2D__LymphoidLeukemia_C91__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()


#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "D0", "D1", "D2", "D3", "D4")
param_values["use_death_as_outcome"] = False

r = PipelineRun("T2D__cancer__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%
param_values["outcome_diagnosis_pattern"] = ("K7")
param_values["use_death_as_outcome"] = False

r = PipelineRun("T2D__liver__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()
#%%
#%%

param_values["outcome_diagnosis_pattern"] = ("G")
param_values["use_death_as_outcome"] = False

r = PipelineRun("T2D__nervous__v14",
                Pipeline("pipeline_tasks\*.py", execution_plan), target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA")
r.execute()


#%%
#%%

from combined_model_report import generate_and_save_combined_model_report, merge_reports

pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"
run_names = [
        "T2D_timesplit__death__v14",
        "T2D_timesplit__MI__v14",
        "T2D_timesplit__CVD_no_HF_MI_ST__v14",
        "T2D_timesplit__CKD_N17-N19__v14",
        "T2D_timesplit__CVD_HF_I50__v14",
        'T2D_timesplit__Stroke__v14',
        "T2D_timesplit_no_weights__CKD_N17-N19__v14",
        "T2D_timesplit__CVD_all__v14",
        "T2D_timesplit__CKD_N17-N19_60days__v14",
        ]
for run_name in run_names:
    generate_and_save_combined_model_report(run_name, pipelines_dir=pipelines_dir)

merge_reports(run_names, "T2D_timesplit__OUTMIX__v14_v15__report", pipelines_dir=pipelines_dir)
#%%

from combined_model_report import generate_and_save_combined_model_report, merge_reports

pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"
run_names = [
        "T2D_timesplit__CKD_N17-N19__v14",
        "T2D_timesplit__CKD_N17-N19_60days__v14",
        ]
for run_name in run_names:
    generate_and_save_combined_model_report(run_name, pipelines_dir=pipelines_dir)

merge_reports(run_names, "T2D_timesplit__BUFF_PERIOD__v14_v15__report", pipelines_dir=pipelines_dir)
#%%


from combined_model_report import generate_and_save_combined_model_report, merge_reports

pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"
run_names = [
        "T2D_notimesplit__death__v14",
        "T2D_notimesplit__MI__v14",
        "T2D_notimesplit__CVD_no_HF_MI_ST__v14",
        "T2D_notimesplit__CKD_N17-N19__v14",
        "T2D_notimesplit__CVD_HF_I50__v14",
        'T2D_notimesplit__Stroke__v14',
        "T2D_notimesplit__CVD_all__v14",
        ]
for run_name in run_names:
    generate_and_save_combined_model_report(run_name, pipelines_dir=pipelines_dir)

merge_reports(run_names, "T2D_notimesplit__OUTMIX__v14_v15__report", pipelines_dir=pipelines_dir)

#%%

from combined_model_report import generate_and_save_combined_model_report, merge_reports

pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"

for time_delta in range(1,5):
    run_names = [
        f"T2D_{time_delta}_timesplit__death__v14",
        f"T2D_{time_delta}_timesplit__MI__v14",
        f"T2D_{time_delta}_timesplit__CVD_no_HF_MI_ST__v14",
        f"T2D_{time_delta}_timesplit__CKD_N17-N19__v14",
        f"T2D_{time_delta}_timesplit__CVD_HF_I50__v14",
        f"T2D_{time_delta}_timesplit__Stroke__v14",
        f"T2D_{time_delta}_timesplit__CVD_all__v14",
        ]
    for run_name in run_names:
        generate_and_save_combined_model_report(run_name, pipelines_dir=pipelines_dir)

    merge_reports(run_names, f"T2D_{time_delta}_timesplit__OUTMIX__v14_v15__report", pipelines_dir=pipelines_dir)
    print("")


#%%

from combined_model_report import generate_and_save_combined_model_report, merge_reports

pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"
run_names = [
        "T2D_notimesplit__CKD_N17-N19__v14",
        "T2D_timesplit_auPRC__CKD_N17-N19__v14",
        "T2D_timesplit__CKD_N17-N19__v14"
        ]
for run_name in run_names:
    generate_and_save_combined_model_report(run_name, pipelines_dir=pipelines_dir)

merge_reports(run_names, "T2D_timesplit_vs_notimesplit_vs_auPRC__OUTMIX__v14_v1__report", pipelines_dir=pipelines_dir)

#%%




#%%

from combined_model_report import generate_and_save_combined_model_report, merge_reports

pipelines_dir="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\"
run_names = [
        "T2D__CVD_I00-I02__v14",
        "T2D__CVD_I05-I09__v14",
        "T2D__CVD_I10-I16__v14",
        "T2D__CVD_I20-I25__v14",
        "T2D__CVD_I26-I28__v14",
        "T2D__CVD_I60-I69__v14"
        ]
for run_name in run_names:
    generate_and_save_combined_model_report(run_name, pipelines_dir=pipelines_dir)

merge_reports(run_names, "T2D__CVDtopcats__v14_report", pipelines_dir=pipelines_dir)
