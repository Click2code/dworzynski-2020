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
import seaborn as sns

from common import load_input, start_stop_timer, my_describe, ready_for_json, dict_to_str

pd.options.display.max_rows = 10

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
t = start_stop_timer

outcome_diagnosis_pattern = ("I") #|| input
use_death_as_outcome = False #|| input

followup_period_years = 5 #|| input


#load_input_run_name = "T2D_timesplit__Stroke__v14"
load_input_run_name = "T2D_timesplit__CKD_N17-N19__v14"
load_input_shared_storage_name = "T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA"
subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input

first_subpop_event_df = load_input(load_input_run_name, "first_subpop_event_df", load_input_shared_storage_name) #|| input
subpop_cpr_df = load_input(load_input_run_name, "subpop_cpr_df", load_input_shared_storage_name) #|| input
subpop_birthplace_df = load_input(load_input_run_name, "subpop_birthplace_df", load_input_shared_storage_name) #|| input
subpop_death_causes_df = load_input(load_input_run_name, "subpop_death_causes_df", load_input_shared_storage_name) #|| input

subpop_sks_df = load_input(load_input_run_name, "subpop_sks_df", load_input_shared_storage_name) #|| input
subpop_diag_df = load_input(load_input_run_name, "subpop_diag_df", load_input_shared_storage_name) #|| input
subpop_prescriptions_df = load_input(load_input_run_name, "subpop_prescriptions_df", load_input_shared_storage_name) #|| input
subpop_ssr_df = load_input(load_input_run_name, "subpop_ssr_df", load_input_shared_storage_name) #|| input

current_addresses_df = load_input(load_input_run_name, "current_addresses_df", load_input_shared_storage_name) #|| input
past_addresses_df = load_input(load_input_run_name, "past_addresses_df", load_input_shared_storage_name) #|| input
past_archive_addresses_df = load_input(load_input_run_name, "past_archive_addresses_df", load_input_shared_storage_name) #|| input

use_parent_diagnoses = False #|| input
use_death_registry = True #|| input
subpop_parents_diag_df = load_input(load_input_run_name, "subpop_parents_diag_df", load_input_shared_storage_name) #|| input

t("Assembling basic feature data frame for subopulation")
subpop_basic_df = pd.merge(subpop_cpr_df[["D_foddato", "C_KON", "dod"]], first_subpop_event_df[["first_subpop_event"]], left_index=True, right_index=True, how="outer")
subpop_basic_df["died_during_followup"] = subpop_basic_df["dod"].notnull() & \
                                                (subpop_basic_df["dod"] < (subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=365*followup_period_years + 30))) & \
                                                (subpop_basic_df["dod"] > (subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=30)))
subpop_basic_df.rename(columns={"D_foddato": "dob", "C_KON": "gender"}, inplace=True)
subpop_basic_df["yob"] = subpop_basic_df["dob"].dt.year
subpop_basic_df.drop(["dob"], axis=1)
subpop_basic_df.loc[subpop_basic_df["gender"] == "K", "gender"] = 0
subpop_basic_df.loc[subpop_basic_df["gender"] == "M", "gender"] = 1
subpop_basic_df["gender"] = subpop_basic_df["gender"].astype(np.int8)
subpop_basic_df = subpop_basic_df.join(subpop_birthplace_df["fodested"], how="left")
subpop_basic_df["fodested"] = subpop_basic_df["fodested"].fillna("Unknown")
subpop_basic_df["fodested"] = subpop_basic_df["fodested"].astype("category")
t("Assembling basic feature data frame for subopulation")
t("Annotating basic feature data frame with outcome information")
columns_to_evaluate_as_first_outcome_dates = []
case_lbnrs_identified_through_deathreg = set()

if len(outcome_diagnosis_pattern) > 0:
    t("Indentifying individuals with outcome diagnosis within subpopulation")
    subpop_diag_outcome_df = subpop_diag_df[subpop_diag_df["C_diag"].str.startswith(outcome_diagnosis_pattern)]
    subpop_first_diag_outcome_df = subpop_diag_outcome_df.sort_values(by="D_inddto", ascending=True).groupby("lbnr").first()
    subpop_first_diag_outcome_df["first_outcome_diag"] = subpop_first_diag_outcome_df["D_inddto"]
    subpop_first_diag_outcome_df = subpop_first_diag_outcome_df[["first_outcome_diag", "C_diag", "C_diagtype"]]
    t("Indentifying individuals with outcome diagnosis within subpopulation")
    
    if use_death_registry:
        subpop_death_outcome_df = subpop_death_causes_df[subpop_death_causes_df["cause"].str.startswith(outcome_diagnosis_pattern)]
        subpop_death_outcome_df = subpop_death_outcome_df[~subpop_death_outcome_df.index.duplicated()]
        
        
        subpop_first_outcome_df = pd.merge(subpop_first_diag_outcome_df, subpop_death_outcome_df, how="outer", left_index=True, right_index=True)
        assert(subpop_first_outcome_df.index.is_unique)
        
        p("num_outcomes_identified_through_lpr_lms", sum(subpop_first_outcome_df["first_outcome_diag"].notnull()))
        p("num_outcomes_identified_through_deathreg", sum(subpop_first_outcome_df["dod"].notnull()))#~
        p("num_outcomes_identified_through_deathreg_and_not_lprlms", sum(subpop_first_outcome_df["first_outcome_diag"].isnull() & subpop_first_outcome_df["dod"].notnull()))
        case_lbnrs_identified_through_deathreg = set(subpop_first_outcome_df[subpop_first_outcome_df["first_outcome_diag"].isnull() & subpop_first_outcome_df["dod"].notnull()].index)
        
        subpop_first_outcome_df.loc[subpop_first_outcome_df["first_outcome_diag"].isnull(), "first_outcome_diag"] = subpop_first_outcome_df.loc[subpop_first_outcome_df["first_outcome_diag"].isnull(), "dod"]
        assert(not subpop_first_outcome_df["first_outcome_diag"].hasnans)
        
        
        subpop_basic_df = subpop_basic_df.join(subpop_first_outcome_df["first_outcome_diag"], how="left")
    else:
        subpop_basic_df = subpop_basic_df.join(subpop_first_diag_outcome_df["first_outcome_diag"], how="left")
    
    columns_to_evaluate_as_first_outcome_dates.append("first_outcome_diag")


if use_death_as_outcome:
    columns_to_evaluate_as_first_outcome_dates.append("dod")


assert((len(outcome_diagnosis_pattern) > 0) | use_death_as_outcome)

subpop_basic_df["first_outcome_date"] = subpop_basic_df.loc[:, columns_to_evaluate_as_first_outcome_dates].min(axis=1, skipna=True)
t("Annotating basic feature data frame with outcome information")

assert(subpop_basic_df.index.is_unique)


t("Annotating basic feature data frame with outcome information")
columns_to_evaluate_as_first_outcome_dates = []
case_lbnrs_identified_through_deathreg = set()

if len(outcome_diagnosis_pattern) > 0:
    t("Indentifying individuals with outcome diagnosis within subpopulation")
    subpop_diag_outcome_df = subpop_diag_df[subpop_diag_df["C_diag"].str.startswith(outcome_diagnosis_pattern)]
    subpop_first_diag_outcome_df = subpop_diag_outcome_df.sort_values(by="D_inddto", ascending=True).groupby("lbnr").first()
    subpop_first_diag_outcome_df["first_outcome_diag"] = subpop_first_diag_outcome_df["D_inddto"]
    subpop_first_diag_outcome_df = subpop_first_diag_outcome_df[["first_outcome_diag", "C_diag", "C_diagtype"]]
    t("Indentifying individuals with outcome diagnosis within subpopulation")
    
    if use_death_registry:
        subpop_death_outcome_df = subpop_death_causes_df[subpop_death_causes_df["cause"].str.startswith(outcome_diagnosis_pattern)]
        subpop_death_outcome_df = subpop_death_outcome_df[~subpop_death_outcome_df.index.duplicated()]
        
        
        subpop_first_outcome_df = pd.merge(subpop_first_diag_outcome_df, subpop_death_outcome_df, how="outer", left_index=True, right_index=True)
        assert(subpop_first_outcome_df.index.is_unique)
        
        p("num_outcomes_identified_through_lpr_lms", sum(subpop_first_outcome_df["first_outcome_diag"].notnull()))
        p("num_outcomes_identified_through_deathreg", sum(subpop_first_outcome_df["dod"].notnull()))#~
        p("num_outcomes_identified_through_deathreg_and_not_lprlms", sum(subpop_first_outcome_df["first_outcome_diag"].isnull() & subpop_first_outcome_df["dod"].notnull()))
        case_lbnrs_identified_through_deathreg = set(subpop_first_outcome_df[subpop_first_outcome_df["first_outcome_diag"].isnull() & subpop_first_outcome_df["dod"].notnull()].index)
        
        subpop_first_outcome_df.loc[subpop_first_outcome_df["first_outcome_diag"].isnull(), "first_outcome_diag"] = subpop_first_outcome_df.loc[subpop_first_outcome_df["first_outcome_diag"].isnull(), "dod"]
        assert(not subpop_first_outcome_df["first_outcome_diag"].hasnans)
        
        
        subpop_basic_df = subpop_basic_df.join(subpop_first_outcome_df["first_outcome_diag"], how="left")
    else:
        subpop_basic_df = subpop_basic_df.join(subpop_first_diag_outcome_df["first_outcome_diag"], how="left")
    
    columns_to_evaluate_as_first_outcome_dates.append("first_outcome_diag")


if use_death_as_outcome:
    columns_to_evaluate_as_first_outcome_dates.append("dod")


assert((len(outcome_diagnosis_pattern) > 0) | use_death_as_outcome)

subpop_basic_df["first_outcome_date"] = subpop_basic_df.loc[:, columns_to_evaluate_as_first_outcome_dates].min(axis=1, skipna=True)
t("Annotating basic feature data frame with outcome information")

assert(subpop_basic_df.index.is_unique)

a = subpop_basic_df[["first_subpop_event","first_outcome_diag"]].dropna(axis=0)
b = (a["first_outcome_diag"] - a["first_subpop_event"]).dt.days

#%%
#sns.distplot(b[(b>0) & (b<200)], kde=False, bins=100)
plt.figure(dpi=200)
ax = sns.distplot(b[(b>-1) & (b<200)], kde=False, bins=100)
ax.set_title("Histogram of days between first T2D diagnosis and first comorbidity diagosis")
ax.set_xlabel("Number of days")
ax.set_ylabel("Number of individuals")



#%%

# Doing the histogram of admission lengths

#%%


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

pd.options.display.max_rows = 10


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
t = start_stop_timer


#load_input_run_name = "T2D_timesplit__Stroke__v14"
load_input_run_name = "T2D_timesplit__CKD_N17-N19__v14"
load_input_shared_storage_name = "T2D__to__OUTCOMES_MIX_v14_1__SHARED_DATA"
subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
lpr_adm_df = load_input(load_input_run_name, "lpr_adm_df", load_input_shared_storage_name) #|| input
lpr_uaf_adm_df = load_input(load_input_run_name, "lpr_uaf_adm_df", load_input_shared_storage_name) #|| input
lpr_diag_df = load_input(load_input_run_name, "lpr_diag_df", load_input_shared_storage_name) #|| input
lpr_uaf_diag_df = load_input(load_input_run_name, "lpr_uaf_diag_df", load_input_shared_storage_name) #|| input

diagnoses_t_diag_types = {"A", "B", "G", "C", "+"} #|| input

t("Extracting subpopulation subset from LPR diagnoses")

subpop_lpr_adm_df = lpr_adm_df.ix[lpr_adm_df["lbnr"].isin(subpop_lbnrs)]
subpop_nonuaf_diag_df = pd.merge(subpop_lpr_adm_df, lpr_diag_df, left_index=True, right_index=True)

e11_diag_df = subpop_nonuaf_diag_df[subpop_nonuaf_diag_df["C_diag"].str.startswith("DE11")]
e11_diag_df = e11_diag_df.sort_values("D_inddto")
e11_first_diag_df = e11_diag_df.groupby("lbnr").first()
a = (e11_first_diag_df["D_uddto"] - e11_first_diag_df["D_inddto"]).dt.days

#%%
plt.figure(dpi=200)
ax = sns.distplot(a[a<100], bins=100,kde=False)
ax.set_title("Histogram of lengths of hospital admissions with first T2D diagnosis")
ax.set_xlabel("Number of days")
ax.set_ylabel("Number of individuals (<100)")
