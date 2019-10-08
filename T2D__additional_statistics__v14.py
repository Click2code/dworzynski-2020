# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:46:47 2019

@author: fskPioDwo
"""

import numpy as np
from itertools import *
from zipfile import ZipFile
import pandas as pd
from os import path
from pprint import pprint, pformat
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
from numpy.random import choice
from patsy import dmatrix
import json

from common import load_input, start_stop_timer, my_describe, ready_for_json, dict_to_str

pd.options.display.max_rows = 30

#%%

#Additional data that i had to recalculate for publication purposes

#Print poulation sizes including followup filtering
from datetime import datetime, timedelta

last_calendar_day_of_data = datetime(2016, 1, 1)
last_day_of_data = last_calendar_day_of_data - timedelta(days=365*5)
for i in range(5):
    load_input_run_name = f"T2D_{i}_timesplit__MI__v14" if i != 0 else "T2D__timesplit__MI__v14"
    load_input_shared_storage_name = f"T2D_{i}__to__OUTCOMES_MIX_v14_1__SHARED_DATA"  if i != 0 else "T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA"
    
    subpop_cpr_df = load_input(load_input_run_name, "subpop_cpr_df", load_input_shared_storage_name) #|| input
    first_subpop_event_df = load_input(load_input_run_name, "first_subpop_event_df", load_input_shared_storage_name) #|| input
    
    filtered_first_subpop_event_df = first_subpop_event_df[first_subpop_event_df["first_subpop_event"] < last_day_of_data]
    filtered_cpr_df = pd.merge(subpop_cpr_df, filtered_first_subpop_event_df, left_index=True, right_index=True, how="right")
    
    print("{}: # {}\t%W {:.3f}\tage {:.3f}".format(i, filtered_cpr_df.shape[0], 100*float(filtered_cpr_df[filtered_cpr_df["C_KON"] == "K"].shape[0])/filtered_cpr_df.shape[0], np.median((filtered_cpr_df["first_subpop_event"] - filtered_cpr_df["D_foddato"]).dt.days)/365.0))
    #print(np.mean(filtered_cpr_df["D_foddato"].dt.year))
    #print(np.mean(filtered_cpr_df["first_subpop_event"].dt.year))
    
    
#Number of categories in places of birth
basic_features_df =  load_input(load_input_run_name, "basic_features_df", load_input_shared_storage_name)
print(basic_features_df["pob"].nunique())

#%%
#Most common drugs prio to exposure
load_input_run_name =  "T2D_timesplit__CVD_no_HF_MI_ST__v14"
load_input_shared_storage_name = "T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA"
prescription_features_df = load_input(load_input_run_name, "prescription_features_df", load_input_shared_storage_name)

#by number of unique individuals taking the drug
r_prescription_features_df = prescription_features_df.where(prescription_features_df == 0, other=1)
print(r_prescription_features_df.sum(axis=0).sort_values(ascending=False)[:10])

#by number of total prescriptions (not reported, just to compare)
print(prescription_features_df.sum(axis=0).sort_values(ascending=False)[:10])
