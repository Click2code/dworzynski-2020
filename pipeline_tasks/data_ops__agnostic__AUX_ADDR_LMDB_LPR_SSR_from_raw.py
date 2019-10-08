# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:39 2016

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
#%%
#TODO just a marker
#||
action_name = "Load LPR raw data"
action_description = ""
action_output = {"lpr_adm_df", "lpr_diag_df", "lpr_uaf_adm_df", "lpr_uaf_diag_df"}
#||

raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input

#Show what files are available in data directory
#
#data_csv_file_paths = list(glob.glob(path.join(raw_data_dir_path, "raw_csv") + "\\*"))
#data_h5_file_paths = list(glob.glob(path.join(raw_data_dir_path, "hdf5") + "\\*"))
#
#print("Files in raw data directory:")
#for csv_fp in data_csv_file_paths:
#    file_name, file_extension = path.split(csv_fp)[1].split(".")
#    print(f"\t {file_name}")


#%%
#Load LPR data

t("Load LPR diagnoses data")
lpr_adm_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_t_adm.csv"), index_col="K_recnum", usecols=["K_recnum", "lbnr", "D_inddto", "D_uddto"])
lpr_adm_df["D_inddto"] = pd.to_datetime(lpr_adm_df["D_inddto"], format="%d%b%Y")
lpr_adm_df["D_uddto"] = pd.to_datetime(lpr_adm_df["D_uddto"], format="%d%b%Y")
lpr_diag_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_t_diag.csv"), index_col="V_recnum", usecols=["V_recnum", "C_diag", "C_diagtype"])
#%%
#The uaf describes visits that are not "finished" - for example when a patient is under continous surveilance - even for multiple years at a time
lpr_uaf_adm_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_uaf_t_adm.csv"), index_col="K_recnum", usecols=["K_recnum", "lbnr", "D_inddto", "D_uddto"])
lpr_uaf_adm_df["D_inddto"] = pd.to_datetime(lpr_uaf_adm_df["D_inddto"], format="%d%b%Y")
lpr_uaf_adm_df["D_uddto"] = pd.to_datetime(lpr_uaf_adm_df["D_uddto"], format="%d%b%Y")
lpr_uaf_diag_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_uaf_t_diag.csv"), index_col="V_recnum", usecols=["V_recnum", "C_diag", "C_diagtype"])
t("Load LPR diagnoses data")


#%%
#TODO just a marker
#||
action_name = "Load CPR data"
action_description = "Load CPR data"
action_output = {"cpr_df", "cpr_birthplace_df"}
#||

raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input

#Load CPR data
t("Load CPR data")
cpr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Cpr3_t_person_lbnr.csv"), index_col="lbnr", usecols=["lbnr","D_foddato","C_KON","lbnrm","lbnrf","C_mor_myn","C_far_myn", "C_STATUS", "D_STATUS_HEN_START"])
cpr_df.loc[:, 'lbnrf'] = cpr_df["lbnrf"].fillna(-1).astype(int)
cpr_df.loc[:, 'lbnrm'] = cpr_df["lbnrm"].fillna(-1).astype(int)
cpr_df.loc[:, "D_foddato"] = pd.to_datetime(cpr_df["D_foddato"], format="%d%b%Y")
cpr_df.loc[cpr_df["C_STATUS"] == 90, "dod"] = pd.to_datetime(cpr_df[cpr_df["C_STATUS"] == 90]["D_STATUS_HEN_START"], format="%d%b%Y")
cpr_birthplace_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Cpr3_t_fodested.csv"), names=["lbnr", "fodested"], index_col="lbnr", header=0)
t("Load CPR data")

#%%
#TODO just a marker
#||
action_name = "Load Birth Register data and identify women and their pregnancy periods"
action_description = "Load Birth Register data and identify women and their pregnancy periods"
action_output = {"pregnancy_dict", "pregnancy_df"}
#||

raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input

cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input

t("Load pregnancy data")
mfr_t_dfoed_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Mfr_t_dfoed.csv"), usecols=["lbnrm","D_FODDTO","V_SVLANGDE"])
mfr_t_dfoed_df.loc[:, "preg_end"] = pd.to_datetime(mfr_t_dfoed_df["D_FODDTO"], format="%d%b%Y")

mfr_t_lfoed_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Mfr_t_lfoed.csv"), usecols=["lbnrm","lbnrb","V_SVLANGDE"])
mfr_mfr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Mfr_mfr.csv"), usecols=["lbnrm","FOEDSELSDATO","GESTATIONSALDER_DAGE"])
mfr_mfr_df.loc[:, "preg_end"] = pd.to_datetime(mfr_mfr_df["FOEDSELSDATO"], format="%d%b%Y")
#%%
mfr_t_lfoed_merged_cpr_df = pd.merge(mfr_t_lfoed_df, cpr_df[["D_foddato"]], left_on="lbnrb", right_index=True, how="left")
mfr_t_lfoed_merged_cpr_df.loc[:, "preg_start"] = mfr_t_lfoed_merged_cpr_df["D_foddato"] - pd.to_timedelta(7 * mfr_t_lfoed_merged_cpr_df["V_SVLANGDE"].fillna(36), unit="D")
mfr_t_lfoed_merged_cpr_df.loc[:, "preg_end"] = mfr_t_lfoed_merged_cpr_df["D_foddato"]
mfr_t_lfoed_merged_cpr_df = mfr_t_lfoed_merged_cpr_df.dropna(subset=["lbnrm"])
mfr_t_lfoed_merged_cpr_df["lbnrm"] = mfr_t_lfoed_merged_cpr_df["lbnrm"].astype(np.int)

mfr_t_dfoed_df.loc[:, "preg_start"] = mfr_t_dfoed_df["preg_end"] - pd.to_timedelta(7 * mfr_t_dfoed_df["V_SVLANGDE"].fillna(36), unit="D")

mfr_mfr_df.loc[:, "preg_start"] = mfr_mfr_df["preg_end"] - pd.to_timedelta(mfr_mfr_df["GESTATIONSALDER_DAGE"], unit="D")
tmp_df = mfr_t_dfoed_df[["lbnrm", "preg_start", "preg_end"]]
tmp_df = tmp_df[tmp_df.isnull().any(axis=1)]
tmp_df.loc[:, "preg_start"] = tmp_df["preg_end"] - pd.to_timedelta(36*7, unit= "D")
mfr_mfr_df.loc[:, "preg_start"]

mfr_mfr_df.loc[mfr_mfr_df["preg_start"].isnull(), "preg_start"] = mfr_mfr_df.loc[mfr_mfr_df["preg_start"].isnull(), "preg_end"] - pd.to_timedelta(36*7, unit= "D")
#%%
pregnancy_dict = {}

for df in [mfr_t_dfoed_df, mfr_t_lfoed_merged_cpr_df, mfr_mfr_df]:
    for e in df.itertuples():
        if e.lbnrm not in pregnancy_dict:
            pregnancy_dict[e.lbnrm] = []
        pregnancy_dict[e.lbnrm].append((e.preg_start, e.preg_end))

pregnancy_df = pd.concat([mfr_mfr_df[["lbnrm", "preg_start", "preg_end"]], mfr_t_lfoed_merged_cpr_df[["lbnrm", "preg_start", "preg_end"]], mfr_t_dfoed_df[["lbnrm", "preg_start", "preg_end"]]])

assert(pregnancy_df[pregnancy_df.isnull().any(axis=1)].shape[0] == 0)

t("Load pregnancy data")

#%%
#TODO just a marker
#||
action_name = "Identify subpopulation individuals from CPR by Xth birthday"
action_description = "Identify subpopulation individuals from CPR"
action_output = {"subpop_lbnrs", "first_subpop_event_df"}
#||

cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input
Xth_birthday = 65 #|| input
Xth_birthday_min_date = date(2000, 1, 1) #|| input
Xth_birthday_max_date = date(2016, 1, 1) #|| input
p("inputs", {
        "Xth_birthday": Xth_birthday,
        "Xth_birthday_min_date": Xth_birthday_min_date,
        "Xth_birthday_max_date": Xth_birthday_max_date})
#we don't strictly need to correct for max date as followup filtering will adjust this but this is measure of decreasing the dataset size

Xth_bd__intermediate_df = pd.DataFrame(
        {"year": cpr_df["D_foddato"].dt.year + Xth_birthday,
         "month": cpr_df["D_foddato"].dt.month,
         "day": cpr_df["D_foddato"].dt.day})

if Xth_birthday % 4: #If it's not a leap year -then fix leaper birth dates
    leapers = (Xth_bd__intermediate_df["month"] == 2) & (Xth_bd__intermediate_df["day"] == 29)
    Xth_bd__intermediate_df.loc[leapers, "month"] = 3
    Xth_bd__intermediate_df.loc[leapers, "day"] = 1


cpr_df["Xth_bd"] = pd.to_datetime(Xth_bd__intermediate_df)

time_filtered_cpr_df = cpr_df[cpr_df["Xth_bd"] > Xth_birthday_min_date]
time_filtered_cpr_df = time_filtered_cpr_df[time_filtered_cpr_df["Xth_bd"] < Xth_birthday_max_date]
time_filtered_cpr_df = time_filtered_cpr_df[(time_filtered_cpr_df["Xth_bd"] < time_filtered_cpr_df["dod"]) | time_filtered_cpr_df["dod"].isnull()]


first_subpop_event_df = pd.DataFrame()
first_subpop_event_df["first_subpop_event"] = time_filtered_cpr_df["Xth_bd"]
subpop_lbnrs = set(time_filtered_cpr_df["Xth_bd"].index)

#%%
#TODO just a marker
#||
action_name = "Identify subpopulation individuals from CPR by specific date"
action_description = "Identify subpopulation individuals from CPR"
action_output = {"subpop_lbnrs", "first_subpop_event_df"}
#||


cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input
t0_date = date(2010, 1, 1) #|| input

p("inputs", {"t0_date": t0_date})

assert(t0_date != None)
assert(isinstance(t0_date, date))

#date is set, the only thing is to make sure that everybody is still alive then
still_alive_cpr_df = cpr_df[(cpr_df["dod"] > t0_date) | pd.isnull(cpr_df["dod"])]
subpop_lbnrs = set(still_alive_cpr_df.index)

still_alive_cpr_df.loc[:, "first_subpop_event_df"] = t0_date
first_subpop_event_df = still_alive_cpr_df.loc[:, "first_subpop_event_df"]

#%%
#TODO just a marker
#||
action_name = "Identify subpopulation individuals from LPR"
action_description = "Identify subpopulation  individuals from LPR based on ICD codes"
action_output = {"subpop_first_subpop_diag_df", "subpop_ICD_lbnrs"}
#||

#Identify individuals who have a given diagnosis
t("Extract subpopulation from LPR data")

subpopulation_icd_code_pattern = "I10" #|| input
drop_subpop_events_during_pregnancies = False #|| input

diagnoses_t_diag_types = {"A", "B", "G", "C", "+"} #|| input

p("inputs", {
        "subpopulation_icd_code_pattern": subpopulation_icd_code_pattern,
        "diagnoses_t_diag_types": diagnoses_t_diag_types,
        "drop_subpop_events_during_pregnancies": drop_subpop_events_during_pregnancies})


lpr_adm_df = load_input(load_input_run_name, "lpr_adm_df", load_input_shared_storage_name) #|| input
lpr_diag_df = load_input(load_input_run_name, "lpr_diag_df", load_input_shared_storage_name) #|| input
lpr_uaf_adm_df = load_input(load_input_run_name, "lpr_uaf_adm_df", load_input_shared_storage_name) #|| input
lpr_uaf_diag_df = load_input(load_input_run_name, "lpr_uaf_diag_df", load_input_shared_storage_name) #|| input
pregnancy_dict = load_input(load_input_run_name, "pregnancy_dict", load_input_shared_storage_name) #|| input

def get_df_of_unique_lbnrs_with_first_occurence_of_certain_code(adm_df, diag_df, icd_code_pattern="E11"):
    if isinstance(icd_code_pattern, str):
        prefixed_icd_code_pattern = "D" + icd_code_pattern
    elif isinstance(icd_code_pattern, (set, list, tuple)):
        prefixed_icd_code_pattern = tuple(["D"+ code for code in icd_code_pattern])
    recnum_set = set(diag_df[diag_df["C_diag"].str.startswith(prefixed_icd_code_pattern) & diag_df["C_diagtype"].isin(diagnoses_t_diag_types)].index)
    reduced_adm_df = adm_df.loc[recnum_set]
    reduced_reduced_adm_df = reduced_adm_df

    num_dropped = 0
    #dropped = []
    if drop_subpop_events_during_pregnancies:
        for e in reduced_adm_df.itertuples():
            if e.lbnr in pregnancy_dict:
                for pregnancy_start, pregnancy_end in pregnancy_dict[e.lbnr]:
                    if ( (e.D_inddto <= pregnancy_start) and (e.D_uddto >= pregnancy_start) ) or \
                        ( (e.D_inddto <= pregnancy_end) and (e.D_uddto >= pregnancy_end) ) or \
                        ( (e.D_inddto >= pregnancy_start) and (e.D_uddto <= pregnancy_end) ):
                            reduced_reduced_adm_df.drop(e.Index)
                            num_dropped+=1
                            #dropped.append(e)

    return reduced_reduced_adm_df.groupby("lbnr", sort=False)["D_uddto"].min(), num_dropped
    #lbnr_set = set(adm_df.ix[recnum_set]["lbnr"])
    #return lbnr_set

first_subpop_ICD_date_series, num_dropped_adm = get_df_of_unique_lbnrs_with_first_occurence_of_certain_code(lpr_adm_df, lpr_diag_df, subpopulation_icd_code_pattern)
subpop_ICD_nonuaf_lbnrs = set(first_subpop_ICD_date_series.index)

first_subpop_ICD_uaf_date_series, num_dropped_uaf_adm = get_df_of_unique_lbnrs_with_first_occurence_of_certain_code(lpr_uaf_adm_df, lpr_uaf_diag_df, subpopulation_icd_code_pattern)
subpop_ICD_uaf_lbnrs = set(first_subpop_ICD_uaf_date_series.index)

#This concatenates UAF and non-UAF series and for each lbnr selects the earlier date if more than one is present
subpop_first_subpop_diag_df = pd.concat([first_subpop_ICD_date_series, first_subpop_ICD_uaf_date_series]).to_frame("first_subpop_ICD_diag")
subpop_first_subpop_diag_df = subpop_first_subpop_diag_df.dropna()
subpop_first_subpop_diag_df = subpop_first_subpop_diag_df.groupby(subpop_first_subpop_diag_df.index, sort=False)["first_subpop_ICD_diag"].min().to_frame("first_subpop_ICD_diag")

subpop_ICD_lbnrs = subpop_ICD_uaf_lbnrs | subpop_ICD_nonuaf_lbnrs

p("Subpopulation diagnoses (ICD) identification", {
        "subpopulation diagnosis pattern": subpopulation_icd_code_pattern,
        "# individuals with the diagnosis from UAF file": len(subpop_ICD_uaf_lbnrs),
        "# individuals with the diagnosis from non-UAF file": len(subpop_ICD_nonuaf_lbnrs),
        "# individuals with diagnosis total": len(subpop_ICD_lbnrs),
        "# subpop events dropped from adm": num_dropped_adm,
        "# subpop events dropped from uaf_adm": num_dropped_uaf_adm
        })

t("Extract subpopulation from LPR data")
#%%


#adm_df=lpr_adm_df
#diag_df=lpr_diag_df
#icd_code_pattern="E11"
#
#if isinstance(icd_code_pattern, str):
#    prefixed_icd_code_pattern = "D" + icd_code_pattern
#elif isinstance(icd_code_pattern, (set, list, tuple)):
#    prefixed_icd_code_pattern = tuple(["D"+ code for code in icd_code_pattern])
#recnum_set = set(diag_df[diag_df["C_diag"].str.startswith(prefixed_icd_code_pattern)].index)
#reduced_adm_df = adm_df.ix[recnum_set]
#
#jdf = reduced_adm_df.join(diag_df, how="b")
#e11_jdf = jdf[jdf["C_diag"].str.startswith(prefixed_icd_code_pattern)]
#
#min_e11_df = e11_jdf.sort_values("D_inddto").groupby(by=["lbnr"]).first()
#
##See:
#min_e11_df[(min_e11_df["D_uddto"] - min_e11_df["D_inddto"]).dt.days > 365]
#
##Also see:
#(min_e11_df["D_uddto"] - min_e11_df["D_inddto"]).dt.days.describe()


#%%
#TODO just a marker
#||
action_name = "Load and identify subpopulation individuals from LMS"
action_description = "Load and identify subpopulation  individuals from LMS based on ATC codes"
action_output = {"subpop_drug_lbnrs", "lms_lbnr_first_subpop_drug_prescriptions_df", "lms_lbnr_subpop_drug_prescriptions_df", "lms_vnr_encryptedcpr_prescriptions_df", "lms_encryptedcpr_to_lbnr_df", "lms_vnr_df"}
#||

#Identify individuals who had a given drug prescription

raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input

t("Extract subpopulation from prescription data")

subpopulation_ATC_code_pattern = "A10B" #|| input
age_restricted_subpopulation_ATC_code_sub_pattern = "A10A" #|| input
subpopulation_min_age_at_first_restricted_presc_event = 30 #|| input
drop_subpop_events_during_pregnancies = True #|| input
pregnancy_dict = load_input(load_input_run_name, "pregnancy_dict", load_input_shared_storage_name) #|| input
pregnancy_df = load_input(load_input_run_name, "pregnancy_df", load_input_shared_storage_name) #|| input
cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input


lms_vnr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "LMS_LAEGEMIDDELOPLYSNINGER.csv"), encoding="latin-1")
all_subpop_drug_vnrs = set(lms_vnr_df.loc[lms_vnr_df["ATC"].str.startswith(subpopulation_ATC_code_pattern).fillna(False) | lms_vnr_df["ATC"].str.startswith(age_restricted_subpopulation_ATC_code_sub_pattern).fillna(False)]["VNR"].tolist())
age_restricted_subpop_drug_vnrs = set(lms_vnr_df.loc[lms_vnr_df["ATC"].str.startswith(age_restricted_subpopulation_ATC_code_sub_pattern).fillna(False)]["VNR"].tolist())

#%%

lms_vnr_encryptedcpr_prescriptions_df = pd.read_table(path.join(raw_data_dir_path, "raw_csv", "LMS_EPIKUR.csv"), sep=",", memory_map=True)

#%%

lms_vnr_encryptedcpr_subpop_drug_prescriptions_df = lms_vnr_encryptedcpr_prescriptions_df.ix[lms_vnr_encryptedcpr_prescriptions_df["VNR"].isin(all_subpop_drug_vnrs)]
lms_vnr_encryptedcpr_subpop_drug_prescriptions_df.loc[:, "EKSD"] = pd.to_datetime(lms_vnr_encryptedcpr_subpop_drug_prescriptions_df["EKSD"], format="%d%b%Y")

lms_encryptedcpr_to_lbnr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "LMS_encrypted_cpr_mapping.csv"), index_col="V_PNR_ENCRYPTED")
lms_lbnr_subpop_drug_prescriptions_df = pd.merge(lms_vnr_encryptedcpr_subpop_drug_prescriptions_df, lms_encryptedcpr_to_lbnr_df, left_on="CPR_ENCRYPTED", right_index=True)


if drop_subpop_events_during_pregnancies:
    pregnant_women_presc_df = lms_lbnr_subpop_drug_prescriptions_df[lms_lbnr_subpop_drug_prescriptions_df["lbnr"].isin(set(pregnancy_dict.keys()))]
    merged_pregnant_women_presc_df = pd.merge(pregnant_women_presc_df[["lbnr", "EKSD"]].reset_index(), pregnancy_df, left_on = "lbnr", right_on="lbnrm")
    merged_pregnant_women_presc_df = merged_pregnant_women_presc_df.set_index("index")
    events_to_delete = merged_pregnant_women_presc_df[(merged_pregnant_women_presc_df["EKSD"] < merged_pregnant_women_presc_df["preg_end"]) & (merged_pregnant_women_presc_df["EKSD"] > merged_pregnant_women_presc_df["preg_start"])]
    indexes_to_delete = events_to_delete.index
    num_presc_removed_due_to_pregnancies = len(indexes_to_delete.unique())
    lms_lbnr_subpop_drug_prescriptions_df = lms_lbnr_subpop_drug_prescriptions_df.drop(indexes_to_delete.unique())
else:
    num_presc_removed_due_to_pregnancies = 0

#filter the lms_lbnr_subpop_drug_prescriptions_df by the prescriptions of sub pattern that occured before a certain birthday
individuals_violating_age_restriction = {}
if age_restricted_subpopulation_ATC_code_sub_pattern and subpopulation_min_age_at_first_restricted_presc_event > 0:
    temp_df = pd.merge(cpr_df[["D_foddato"]], lms_lbnr_subpop_drug_prescriptions_df, left_index=True, right_on="lbnr")
    prescriptions_before_Xth_birthday = temp_df[(temp_df["EKSD"] - temp_df["D_foddato"]).dt.days/365 < subpopulation_min_age_at_first_restricted_presc_event]
    prescriptions_for_agefiltered_drugs_before_Xth_birthday = prescriptions_before_Xth_birthday[prescriptions_before_Xth_birthday["VNR"].isin(age_restricted_subpop_drug_vnrs)]
    lnbrs_with_agefiltered_drugs_before_Xth_birthday = set(prescriptions_for_agefiltered_drugs_before_Xth_birthday["lbnr"])
    agefiltered_df = temp_df[~temp_df["lbnr"].isin(lnbrs_with_agefiltered_drugs_before_Xth_birthday)]
    agefiltered_df = agefiltered_df.drop("D_foddato", axis=1)
    num_presc_removed_due_to_agefilter = len(lms_lbnr_subpop_drug_prescriptions_df) - len(agefiltered_df)
    lms_lbnr_subpop_drug_prescriptions_df = agefiltered_df
else:
    num_presc_removed_due_to_agefilter = 0
    lnbrs_with_agefiltered_drugs_before_Xth_birthday = set()

# The problem with thisis that if we remove events for these individuals they will still count as T2Ds, i should drop the individuals entirely. Need to decide whether to use A10A AND A10X (what is A10X?)


lms_lbnr_first_subpop_drug_prescriptions_df = lms_lbnr_subpop_drug_prescriptions_df.groupby("lbnr", sort=False)["EKSD"].min().to_frame("first_subpop_drug_presc_date")

# Now drop individuals for whom the first prescription occured before age of X
# We drop these individuals completely as they might have Type 1 Diabetes

#if subpopulation_min_age_at_first_presc_event > 0:
#    lms_lbnr_first_subpop_drug_presc_and_birthday_df = pd.merge(cpr_df[["D_foddato"]], lms_lbnr_first_subpop_drug_prescriptions_df, left_index=True, right_index=True)
#    lbnrs_with_first_diag_after_30th_bday = lms_lbnr_first_subpop_drug_presc_and_birthday_df[(lms_lbnr_first_subpop_drug_presc_and_birthday_df["first_subpop_drug_presc_date"] - lms_lbnr_first_subpop_drug_presc_and_birthday_df["D_foddato"]).dt.days/365 > subpopulation_min_age_at_first_presc_event].index
#    num_individuals_with_first_prescription_before_age_of_X = len(lms_lbnr_first_subpop_drug_prescriptions_df) - len(lbnrs_with_first_diag_after_30th_bday)
#    lms_lbnr_first_subpop_drug_prescriptions_df = lms_lbnr_first_subpop_drug_prescriptions_df.loc[lbnrs_with_first_diag_after_30th_bday]
#else:
#    num_individuals_with_first_prescription_before_age_of_X = 0


subpop_drug_lbnrs = set(lms_lbnr_first_subpop_drug_prescriptions_df.index)

t("Extract subpopulation from prescription data")

p("Subpopulation prescriptions (ATC) identification", {
        "prescription pattern": subpopulation_ATC_code_pattern,
        "# individuals with prescription": len(lms_lbnr_subpop_drug_prescriptions_df["lbnr"].unique()),
        "# number of events removed due to pregnancy": num_presc_removed_due_to_pregnancies,
        "# individuals with age restricted prescription before age threshold": num_presc_removed_due_to_agefilter,
        "# individuals removed due to presc of age restricted drug before age limit": len(lnbrs_with_agefiltered_drugs_before_Xth_birthday)
        })


#%%
#||
action_name = "Remove females below 40 yo"
action_description = "Remove females below 40 yo"
action_output = {"femfiltered_subpop_drug_lbnrs", "femfiltered_lms_lbnr_first_subpop_drug_prescriptions_df"}
#||

# If T2D remove women above 30 (as metfomin could be prescribed to pregnant women)

cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input
subpop_drug_lbnrs = load_input(load_input_run_name, "subpop_drug_lbnrs", load_input_shared_storage_name) #|| input
lms_lbnr_subpop_drug_prescriptions_df = load_input(load_input_run_name, "lms_lbnr_subpop_drug_prescriptions_df", load_input_shared_storage_name) #|| input

min_female_age = 40 #|| input


filtering_out_women_intermediate_df = pd.merge(cpr_df[["D_foddato", "C_KON"]], lms_lbnr_subpop_drug_prescriptions_df[["EKSD", "lbnr"]], left_index=True, right_on="lbnr")

men_selector = filtering_out_women_intermediate_df["C_KON"] == "M"
female_selector = filtering_out_women_intermediate_df["C_KON"] == "K"
older_than_selector = lambda X_years: (filtering_out_women_intermediate_df["EKSD"] - filtering_out_women_intermediate_df["D_foddato"]).dt.days /365 > X_years

filtering_out_women_intermediate_df = filtering_out_women_intermediate_df[men_selector | (female_selector & older_than_selector(min_female_age))]

femfiltered_lms_lbnr_first_subpop_drug_prescriptions_df = filtering_out_women_intermediate_df[["EKSD", "lbnr"]].groupby("lbnr", sort=False)["EKSD"].min().to_frame("first_subpop_drug_presc_date")

femfiltered_subpop_drug_lbnrs = set(femfiltered_lms_lbnr_first_subpop_drug_prescriptions_df.index)

p("Subpopulation prescriptions (ATC) identification", {
        "min_female_age": min_female_age,
        "# individuals before female filtering": len(subpop_drug_lbnrs),
        "# individuals after female filtering": len(femfiltered_subpop_drug_lbnrs)
        })

#%%
#TODO just a marker
#||
action_name = "Merge subpopulation LBNRs and first event dataframes identified in LMS and LPR"
action_description = ""
action_output = {"subpop_lbnrs", "first_subpop_event_df"}
#||

subpopulation_days_offset = 0 #|| input
earliest_required_year_of_diagnosis = 1995 #|| input
subpop_ICD_lbnrs = load_input(load_input_run_name, "subpop_ICD_lbnrs", load_input_shared_storage_name) #|| input
subpop_drug_lbnrs = load_input(load_input_run_name, "subpop_drug_lbnrs", load_input_shared_storage_name) #|| input
subpop_first_subpop_diag_df = load_input(load_input_run_name, "subpop_first_subpop_diag_df", load_input_shared_storage_name) #|| input
lms_lbnr_first_subpop_drug_prescriptions_df = load_input(load_input_run_name, "lms_lbnr_first_subpop_drug_prescriptions_df", load_input_shared_storage_name) #|| input


first_subpop_event_df = pd.merge(subpop_first_subpop_diag_df, lms_lbnr_first_subpop_drug_prescriptions_df, left_index=True, right_index=True, how="outer")
first_subpop_event_df["first_subpop_event"] = first_subpop_event_df.min(axis=1, skipna=True)
first_subpop_event_df["first_subpop_event"] += pd.to_timedelta(subpopulation_days_offset, unit="D")

#%%
first_subpop_event_year_nulls = first_subpop_event_df["first_subpop_event"].dt.year
first_subpop_event_year = first_subpop_event_year_nulls[first_subpop_event_year_nulls.notnull()]
first_subpop_event_year_counts = list(first_subpop_event_year.value_counts().to_dict().items())
first_subpop_event_year_counts = sorted(first_subpop_event_year_counts, key=lambda x: x[0])
fig, ax = plt.subplots()
ax.bar([year for year, count in first_subpop_event_year_counts], [count for year, count in first_subpop_event_year_counts])
ax.set_xticklabels([year for year, count in first_subpop_event_year_counts])
ax.set_xticks([year for year, count in first_subpop_event_year_counts])
plt.xticks(rotation=90)
#|| savefig first_subpop_event_year
#%%
#drop individuals who had their first diagnosis before 2000
first_subpop_event_df = first_subpop_event_df[first_subpop_event_df["first_subpop_event"].dt.year >= earliest_required_year_of_diagnosis]
subpop_lbnrs = set(first_subpop_event_df.index)


#%%

p("Final count of individuals in subpopulation", {
        "num individuals identified by ICD codes": len(subpop_ICD_lbnrs),
        "num individuals identified by ATC codes": len(subpop_drug_lbnrs),
        "overlap between above": len(subpop_ICD_lbnrs & subpop_drug_lbnrs),
        "total number of identified individauls (sum of above)": len(subpop_lbnrs),
        "num dates which are null": sum(first_subpop_event_df["first_subpop_event"].isnull())
        })

#%%
#TODO just a marker
#Now that we have identified subpop patients we extract them from both LPR and LMS

#||
action_name = "Extract subpopulation from LPR"
action_description = ""
action_output = {"subpop_diag_df", "subpop_lpr_adm_df", "subpop_lpr_uaf_adm_df"}
#||

subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
lpr_adm_df = load_input(load_input_run_name, "lpr_adm_df", load_input_shared_storage_name) #|| input
lpr_uaf_adm_df = load_input(load_input_run_name, "lpr_uaf_adm_df", load_input_shared_storage_name) #|| input
lpr_diag_df = load_input(load_input_run_name, "lpr_diag_df", load_input_shared_storage_name) #|| input
lpr_uaf_diag_df = load_input(load_input_run_name, "lpr_uaf_diag_df", load_input_shared_storage_name) #|| input

diagnoses_t_diag_types = {"A", "B", "G", "C", "+"} #|| input

t("Extracting subpopulation subset from LPR diagnoses")

subpop_lpr_adm_df = lpr_adm_df.ix[lpr_adm_df["lbnr"].isin(subpop_lbnrs)]
subpop_nonuaf_diag_df = pd.merge(subpop_lpr_adm_df, lpr_diag_df, left_index=True, right_index=True)

subpop_lpr_uaf_adm_df = lpr_uaf_adm_df.ix[lpr_uaf_adm_df["lbnr"].isin(subpop_lbnrs)]
subpop_uaf_diag_df = pd.merge(subpop_lpr_uaf_adm_df, lpr_uaf_diag_df, left_index=True, right_index=True)

subpop_diag_df = pd.concat([subpop_nonuaf_diag_df, subpop_uaf_diag_df])
subpop_diag_df.reset_index(inplace=True, drop=True)

#Drop the letter D prefix from the diagnosis code
subpop_diag_df["C_diag"] =  subpop_diag_df["C_diag"].str.slice(1,None,None)

#Drop diagnoses of other diagtypes than those from diagnoses_t_diag_types
subpop_diag_df = subpop_diag_df[subpop_diag_df["C_diagtype"].isin(diagnoses_t_diag_types)]

#Drop diagnoses not starting with a letter
subpop_diag_df = subpop_diag_df[subpop_diag_df["C_diag"].str.contains("^[a-zA-Z]")]

#Drop duplicates
subpop_diag_df = subpop_diag_df.drop_duplicates()

t("Extracting subpopulation subset from LPR diagnoses")

#%%

#||
action_name = "Extract subpopulation parents from LPR"
action_description = ""
action_output = {"subpop_parents_lbnrs", "subpop_parents_diag_df", "subpop_parents_lpr_adm_df", "subpop_parents_lpr_uaf_adm_df"}
#||

subpop_cpr_df = load_input(load_input_run_name, "subpop_cpr_df", load_input_shared_storage_name) #|| input
lpr_adm_df = load_input(load_input_run_name, "lpr_adm_df", load_input_shared_storage_name) #|| input
lpr_uaf_adm_df = load_input(load_input_run_name, "lpr_uaf_adm_df", load_input_shared_storage_name) #|| input
lpr_diag_df = load_input(load_input_run_name, "lpr_diag_df", load_input_shared_storage_name) #|| input
lpr_uaf_diag_df = load_input(load_input_run_name, "lpr_uaf_diag_df", load_input_shared_storage_name) #|| input

diagnoses_t_diag_types = {"A", "B", "G", "C", "+"} #|| input

t("Extracting subpopulation parents subset from LPR diagnoses")
#subpop_cpr_df = cpr_df[cpr_df.index.isin(subpop_lbnrs)]
#subpop_cpr_df['lbnrm'] = subpop_cpr_df["lbnrm"].fillna(-1).astype(int)
#subpop_cpr_df['lbnrf'] = subpop_cpr_df["lbnrf"].fillna(-1).astype(int)
subpop_parents_lbnrs = set(subpop_cpr_df['lbnrm'].values) | set(subpop_cpr_df['lbnrf'].values)
subpop_parents_lbnrs.remove(-1)

subpop_parents_lpr_adm_df = lpr_adm_df.loc[lpr_adm_df["lbnr"].isin(subpop_parents_lbnrs)]
subpop_parents_nonuaf_diag_df = pd.merge(subpop_parents_lpr_adm_df, lpr_diag_df, left_index=True, right_index=True)

subpop_parents_lpr_uaf_adm_df = lpr_uaf_adm_df.loc[lpr_uaf_adm_df["lbnr"].isin(subpop_parents_lbnrs)]
subpop_parents_uaf_diag_df = pd.merge(subpop_parents_lpr_uaf_adm_df, lpr_uaf_diag_df, left_index=True, right_index=True)

subpop_parents_diag_df = pd.concat([subpop_parents_nonuaf_diag_df, subpop_parents_uaf_diag_df])
subpop_parents_diag_df.reset_index(inplace=True, drop=True)

#Drop the letter D prefix from the diagnosis code
subpop_parents_diag_df["C_diag"] =  subpop_parents_diag_df["C_diag"].str.slice(1,None,None)

#Drop diagnoses of other diagtypes than those from diagnoses_t_diag_types
subpop_parents_diag_df = subpop_parents_diag_df[subpop_parents_diag_df["C_diagtype"].isin(diagnoses_t_diag_types)]

subpop_parents_diag_df = subpop_parents_diag_df.rename(columns={"lbnr": "parent_lbnr"})

#Drop diagnoses not starting with a letter
#subpop_parents_diag_df = subpop_parents_diag_df[subpop_parents_diag_df["C_diag"].str.contains("^[a-zA-Z]")]


#This reindexes the parent dataframe so that it contains reference to lbnr or the child
subpop_cpr_df["lbnr"] = subpop_cpr_df.index
subpop_parents_diag_df_inter = pd.merge(subpop_parents_diag_df, subpop_cpr_df[["lbnr", "lbnrm"]], how="left", left_on="parent_lbnr", right_on="lbnrm")
subpop_parents_diag_df_inter = subpop_parents_diag_df_inter.rename(columns={"lbnr": "child_lbnr_1"})
subpop_parents_diag_df_inter = pd.merge(subpop_parents_diag_df_inter, subpop_cpr_df[["lbnr", "lbnrf"]], how="left", left_on="parent_lbnr", right_on="lbnrf")
subpop_parents_diag_df_inter = subpop_parents_diag_df_inter.rename(columns={"lbnr": "child_lbnr_2"})
subpop_parents_diag_df_inter.loc[:, "child_lbnr"] = subpop_parents_diag_df_inter["child_lbnr_1"]
subpop_parents_diag_df_inter["child_lbnr"] = subpop_parents_diag_df_inter["child_lbnr"].where(subpop_parents_diag_df_inter["child_lbnr"].notnull(), other=subpop_parents_diag_df_inter["child_lbnr_2"])
del subpop_parents_diag_df_inter["child_lbnr_1"]
del subpop_parents_diag_df_inter["child_lbnr_2"]
subpop_parents_diag_df = subpop_parents_diag_df_inter.drop_duplicates()

t("Extracting subpopulation parents subset from LPR diagnoses")
#%%
#TODO just a marker
#||
action_name = "Extract subpopulation from LMS"
action_description = ""
action_output = {"subpop_prescriptions_df"}
#||
raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input
subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
#lms_vnr_encryptedcpr_prescriptions_df = load_input(load_input_run_name, "lms_vnr_encryptedcpr_prescriptions_df") input
#lms_encryptedcpr_to_lbnr_df = load_input(load_input_run_name, "lms_encryptedcpr_to_lbnr_df") input
#lms_vnr_df = load_input(load_input_run_name, "lms_vnr_df") input

t("Load LMS static data")
lms_vnr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "LMS_LAEGEMIDDELOPLYSNINGER.csv"), encoding="latin-1")
lms_vnr_encryptedcpr_prescriptions_df = pd.read_table(path.join(raw_data_dir_path, "raw_csv", "LMS_EPIKUR.csv"), sep=",", memory_map=True)
lms_encryptedcpr_to_lbnr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "LMS_encrypted_cpr_mapping.csv"), index_col="V_PNR_ENCRYPTED")
t("Load LMS static data")


t("Extracting subpopulation subset from LMS (prescriptions)")
subpop_encryptedcprs = set(lms_encryptedcpr_to_lbnr_df.ix[lms_encryptedcpr_to_lbnr_df["lbnr"].isin(subpop_lbnrs)].index)
subpop_vnr_encryptedcpr_prescriptions_df = lms_vnr_encryptedcpr_prescriptions_df.ix[lms_vnr_encryptedcpr_prescriptions_df["CPR_ENCRYPTED"].isin(subpop_encryptedcprs)]
subpop_vnr_lbnr_prescriptions_df = pd.merge(subpop_vnr_encryptedcpr_prescriptions_df, lms_encryptedcpr_to_lbnr_df, left_on="CPR_ENCRYPTED", right_index=True)
subpop_vnr_lbnr_prescriptions_df.drop("CPR_ENCRYPTED", 1, inplace=True)
subpop_vnr_lbnr_prescriptions_df.reset_index(inplace=True, drop=True)

subpop_prescriptions_df = pd.merge(subpop_vnr_lbnr_prescriptions_df, lms_vnr_df[["ATC", "VNR"]], left_on="VNR", right_on="VNR")
subpop_prescriptions_df["EKSD"] = pd.to_datetime(subpop_prescriptions_df["EKSD"], format="%d%b%Y")
t("Extracting subpopulation subset from LMS (prescriptions)")

#%%
#TODO just a marker
#||
action_name = "Extract subpopulation from CPR"
action_description = ""
action_output = {"subpop_cpr_df", "subpop_birthplace_df", }
#||

subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input


cpr_birthplace_df = load_input(load_input_run_name, "cpr_birthplace_df", load_input_shared_storage_name) #|| input

t("Extracting subpopulation subset from CPR")

subpop_cpr_df = cpr_df.loc[cpr_df.index.isin(subpop_lbnrs)]
subpop_birthplace_df = cpr_birthplace_df.loc[cpr_birthplace_df.index.isin(subpop_lbnrs)]

t("Extracting subpopulation subset from CPR")
#%%

#TODO just a marker
#||
action_name = "Load Death register data and extract subpopulation"
action_description = "Load Death register data and extract subpopulation"
action_output = {"subpop_dod_dict", "subpop_death_causes_df"}
#||

raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input
subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input

#Load dod data
t("Load Death register data and extract subpopulation")

d1_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Dar_t_dodsaarsag_1.csv"), index_col="lbnr", usecols=["lbnr","C_DOD1", "C_DOD2", "C_DOD3", "C_DOD4"])
d2_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Dar_t_dodsaarsag_2.csv"), index_col="lbnr", usecols=["lbnr", "C_DODTILGRUNDL_ACME", "C_DOD_1A", "C_DOD_1B", "C_DOD_1C", "C_DOD_1D", "C_DOD_21", "C_DOD_22", "C_DOD_23", "C_DOD_24", "C_DOD_25", "C_DOD_26", "C_DOD_27", "C_DOD_28"])

d1_df = d1_df.loc[d1_df.index.isin(subpop_lbnrs)]
d2_df = d2_df.loc[d2_df.index.isin(subpop_lbnrs)]


death_causes_1_df = pd.concat([d1_df[[col]].rename(columns={col: "cause"}) for col in d1_df.columns], axis=0, join="outer")
death_causes_2_df = pd.concat([d2_df[[col]].rename(columns={col: "cause"}) for col in d2_df.columns], axis=0, join="outer")
subpop_death_causes_df = pd.concat([death_causes_1_df, death_causes_2_df], axis=0, join="outer").dropna()
subpop_death_causes_df = subpop_death_causes_df.join(cpr_df["dod"])

subpop_dod_dict = {lbnr: set(death_causes.dropna()) for lbnr, death_causes in d1_df.iterrows()} #Yes, this is terrible but it works and is fast enough
subpop_dod_dict.update({lbnr: set(death_causes.dropna()) for lbnr, death_causes in d2_df.iterrows()})

t("Load Death register data and extract subpopulation")

#%%
#TODO just a marker
#Load address data
# Note that this data needs to be subsetted so that it doesn't contain information after t0

#||
action_name = "Extract subpopulation from Address data"
action_description = ""
action_output = {"current_addresses_df", "past_addresses_df", "past_archive_addresses_df"}
#||

subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input

t("Loading and subsetting address data")

current_addresses_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Cpr3_t_adresse.csv"), index_col="lbnr")
current_addresses_df = current_addresses_df.ix[current_addresses_df.index.isin(subpop_lbnrs)]
current_addresses_df["D_TILFLYT_DATO"] = pd.to_datetime(current_addresses_df["D_TILFLYT_DATO"], format="%d%b%Y")
past_addresses_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Cpr3_t_adresse_hist.csv"), index_col="lbnr")
past_addresses_df = past_addresses_df.ix[past_addresses_df.index.isin(subpop_lbnrs)]
past_addresses_df["D_TILFLYT_DATO"] = pd.to_datetime(past_addresses_df["D_TILFLYT_DATO"], format="%d%b%Y")
past_addresses_df["D_FRAFLYT_DATO"] = pd.to_datetime(past_addresses_df["D_FRAFLYT_DATO"], format="%d%b%Y")
past_archive_addresses_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Cpr3_t_arkiv_adresse_hist.csv"), index_col="lbnr")
past_archive_addresses_df = past_archive_addresses_df.ix[past_archive_addresses_df.index.isin(subpop_lbnrs)]
past_archive_addresses_df["D_TILFLYT_DATO"] = pd.to_datetime(past_archive_addresses_df["D_TILFLYT_DATO"], format="%d%b%Y")
past_archive_addresses_df["D_FRAFLYT_DATO"] = pd.to_datetime(past_archive_addresses_df["D_FRAFLYT_DATO"], format="%d%b%Y")

t("Loading and subsetting address data")

#%%


#%%
#TODO just a marker
#||
action_name = "Extract subpopulation from LPR Procedures"
action_description = ""
action_output = {"subpop_sks_df"}
#||

subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
subpop_lpr_uaf_adm_df = load_input(load_input_run_name, "subpop_lpr_uaf_adm_df", load_input_shared_storage_name) #|| input
subpop_lpr_adm_df = load_input(load_input_run_name, "subpop_lpr_adm_df", load_input_shared_storage_name) #|| input
raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input

#Operations
#Operations are divided into 3 files
#   Lpr_t_sksopr.csv - SKS codes for operations 1996->
#   Lpr_t_opr.csv - some other type of codes for operations ->1996
#   Lpr_t_sksube.csv - some other operations (non surgical treatments) 1999->

t("Loading and subsetting LPR procedures data")

lpr_sksopr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_t_sksopr.csv"), index_col="V_recnum", usecols=["V_recnum", "C_opr", "D_odto"])
lpr_sksube_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_t_sksube.csv"), index_col="V_recnum", usecols=["V_recnum", "C_opr", "D_odto"])
lpr_non_uaf_sks_df = pd.concat([lpr_sksopr_df, lpr_sksube_df])

lpr_uaf_sksopr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_uaf_t_sksopr.csv"), index_col="V_recnum", usecols=["V_recnum", "C_opr", "D_odto"])
lpr_uaf_sksube_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "Lpr_uaf_t_sksube.csv"), index_col="V_recnum", usecols=["V_recnum", "C_opr", "D_odto"])
lpr_uaf_sks_df = pd.concat([lpr_uaf_sksopr_df, lpr_uaf_sksube_df])

subpop_uaf_sks_df = pd.merge(subpop_lpr_uaf_adm_df, lpr_uaf_sks_df, left_index=True, right_index=True)
subpop_nonuaf_sks_df = pd.merge(subpop_lpr_adm_df, lpr_non_uaf_sks_df, left_index=True, right_index=True)

subpop_sks_df = pd.concat([subpop_uaf_sks_df, subpop_nonuaf_sks_df])

t("Loading and subsetting LPR procedures data")

#%%
#TODO just a marker
#SSR
#||
action_name = "Extract subpopulation from SSR"
action_description = ""
action_output = {"subpop_ssr_df"}
#||

subpop_lbnrs = load_input(load_input_run_name, "subpop_lbnrs", load_input_shared_storage_name) #|| input
raw_data_dir_path = "V:\\Projekter\\FSEID00001620\\Piotr\\Data\\" #|| input

t("Loading and subsetting SSR data")

ssr_df = pd.read_csv(path.join(raw_data_dir_path, "raw_csv", "ssr_t_ssik.csv"), index_col="lbnr", usecols=["lbnr","V_HONUGE","C_SPECIALE","V_ANTYDEL","C_YDELSESNR","V_KONTAKT","V_ALDER","year"])
subpop_ssr_df = ssr_df.ix[ssr_df.index.isin(subpop_lbnrs)]
subpop_ssr_df["date"] = pd.to_datetime(subpop_ssr_df["year"].astype(str) + "-W" + subpop_ssr_df["V_HONUGE"].astype(str) + "-1", format="%Y-W%W-%w")
subpop_ssr_df.drop(["V_HONUGE", "year"], axis=1, inplace=True)
subpop_ssr_df["C_YDELSESNR"] = subpop_ssr_df["C_YDELSESNR"].astype(str)
subpop_ssr_df["C_SPECIALE"] = subpop_ssr_df["C_SPECIALE"].astype(str)
subpop_ssr_df = subpop_ssr_df.reset_index()

t("Loading and subsetting SSR data")

#%%
#subpop_drug_lbnrs
#subpop_ICD_lbnrs
#subpop_lbnrs
#
#first_subpop_event_df
#subpop_cpr_df
#subpop_birthplace_df
#subpop_sks_df
#subpop_diag_df
#subpop_prescriptions_df

#%%

#TODO:
#    * think about address data - keep it simple - place of birth or last one before onset - use above
#    * add psych info?

#%%
#TODO: just a marker
#||
action_name = "Filter events and individuals"
action_description = "Applies filtering to events (pre diagnosis and post diagnosis time window, support requirement) and individuals (required followup)"
#action_output = {"counts_dataset", "multihot_dataset", "dataset_common"}
#||
#%%

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

buffer_period_length = 30 #|| input

#%%

#Create a dataframe of first outcome occurence (if any) for each individual
#outcome_diagnosis_pattern = ("I20", "I21" "I22", "I23", "I24", "I25", "I61", "I63", "I64", "I679") #|| input
outcome_diagnosis_pattern = ("I61", "I62", "I63", "I64") #|| input
use_death_as_outcome = False #|| input

followup_period_years = 5 #|| input

#p("inputs", {
#        "outcome_diagnosis_pattern": outcome_diagnosis_pattern,
#        "use_death_as_outcome": use_death_as_outcome,
#        "last_calendar_day_of_data": last_calendar_day_of_data,
#        "followup_period_years": followup_period_years,
#        "pre_t0_time_window": pre_t0_time_window,
#        "post_t0_time_window": post_t0_time_window,
#        "use_diag_codes_up_to_length": use_diag_codes_up_to_length,
#        "use_proc_codes_up_to_length": use_proc_codes_up_to_length,
#        "use_drug_codes_up_to_length": use_drug_codes_up_to_length,
#        "diag_feature_min_support": diag_feature_min_support,
#        "sksopr_feature_min_support": sksopr_feature_min_support,
#        "prescription_feature_min_support": prescription_feature_min_support,
#        "outpatient_visit_feature_min_support": outpatient_visit_feature_min_support,})


#%%

t("Assembling basic feature data frame for subopulation")
subpop_basic_df = pd.merge(subpop_cpr_df[["D_foddato", "C_KON", "dod"]], first_subpop_event_df[["first_subpop_event"]], left_index=True, right_index=True, how="outer")
subpop_basic_df["died_during_followup"] = subpop_basic_df["dod"].notnull() & \
                                                (subpop_basic_df["dod"] < (subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=365*followup_period_years + buffer_period_length))) & \
                                                (subpop_basic_df["dod"] > (subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=buffer_period_length)))
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


#%%


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



#%%

p("Pre-filtering", {
        "# individuals": subpop_basic_df.shape[0],
        "# individuals with outcome": subpop_basic_df[subpop_basic_df["first_outcome_date"].notnull()].shape[0],
        "year of birth description": my_describe(subpop_basic_df["yob"]),
        "% women": 100.0 * sum(subpop_basic_df["gender"] == 0) / subpop_basic_df["gender"].shape[0],
        "age of first subpop event description": my_describe((subpop_basic_df["first_subpop_event"] - subpop_basic_df["dob"]).dt.days/365),
        "year of first subpop event description": my_describe(subpop_basic_df["first_subpop_event"].dt.year),
        "time difference between subpop event and outcome event description": my_describe((subpop_basic_df["first_outcome_date"] - subpop_basic_df["first_subpop_event"]).dt.days),
        "time difference between subpop event and death event description": my_describe((subpop_basic_df["dod"] - subpop_basic_df["first_subpop_event"]).dt.days)
})

#%%
def get_counts_of_events_per_individual(event_df, unique_cols, event_code_col):
    edf = event_df[unique_cols].drop_duplicates()
    agg = edf.groupby("lbnr").agg("count")
    num_unique_codes = edf[event_code_col].nunique()
    return agg[event_code_col], num_unique_codes

#modified_subpop_ssr_df = subpop_ssr_df
#modified_subpop_ssr_df["visit_code"] = modified_subpop_ssr_df["C_SPECIALE"]*(10**5) + modified_subpop_ssr_df["C_YDELSESNR"]

pre_filtering_ssr_counts, pre_filtering_ssr_nunique = get_counts_of_events_per_individual(subpop_ssr_df, ["lbnr", "date", "C_SPECIALE", "C_YDELSESNR"], "C_YDELSESNR")
pre_filtering_diag_counts, pre_filtering_diag_nunique = get_counts_of_events_per_individual(subpop_diag_df, ["lbnr", "D_inddto", "C_diag"], "C_diag")
pre_filtering_procedure_counts, pre_filtering_procedure_nunique = get_counts_of_events_per_individual(subpop_sks_df, ["lbnr", "D_inddto", "C_opr"], "C_opr")
pre_filtering_presc_counts, pre_filtering_presc_nunique = get_counts_of_events_per_individual(subpop_prescriptions_df, ["lbnr", "VNR", "EKSD"], "VNR")

p("Pre-filtering feature counts", {
        "SSR": my_describe(pre_filtering_ssr_counts),
        "sum SSR": pre_filtering_ssr_counts.sum(),
        "# unique SSR": pre_filtering_ssr_nunique,
        "Diagnoses": my_describe(pre_filtering_diag_counts),
        "sum Diagnoses": pre_filtering_diag_counts.sum(),
        "# unique Diagnoses": pre_filtering_diag_nunique,
        "Prescriptions": my_describe(pre_filtering_presc_counts),
        "sum Prescriptions": pre_filtering_presc_counts.sum(),
        "# unique Prescriptions": pre_filtering_presc_nunique,
        "Procedures": my_describe(pre_filtering_procedure_counts),
        "sum Procedures": pre_filtering_procedure_counts.sum(),
        "# unique Procedures": pre_filtering_procedure_nunique
        })


#%%

#%%

# Sanity filtering, filtering out individuals

def filter_out_individuals_based_on_basic_df(condition, filter_name, original_lbnr_set):
    filtered_lbnr_set = set(subpop_basic_df[condition].index)
    p(f"Filtering results for {filter_name}", {
            "# individuals pre filtering": len(original_lbnr_set),
            "# individuals post filtering": len(original_lbnr_set - filtered_lbnr_set),
            "% reduction": 100.0 * (len(filtered_lbnr_set))/len(original_lbnr_set)
            })
    return original_lbnr_set - filtered_lbnr_set


t("Filtering out individuals with outcome events before dob")
#Filter out individuals who have their outcome earlier than their birth date
sanity_filtered_subpop_lbnr_set = filter_out_individuals_based_on_basic_df(subpop_basic_df["first_subpop_event"] < subpop_basic_df["dob"], "first subpop diag before birth date", set(subpop_basic_df.index))
sanity_filtered_subpop_lbnr_set = filter_out_individuals_based_on_basic_df(subpop_basic_df["first_outcome_date"] < subpop_basic_df["dob"], "first outcome diag before birth date", sanity_filtered_subpop_lbnr_set)
t("Filtering out individuals with outcome events before dob")


#%%

t("Filtering out individuals with subpop events after dod")
#Filter out individuals who have their outcome earlier than their birth date
sanity_filtered_subpop_lbnr_set = filter_out_individuals_based_on_basic_df(subpop_basic_df["first_subpop_event"] > subpop_basic_df["dod"], "first subpop diag after death  date", sanity_filtered_subpop_lbnr_set)
sanity_filtered_subpop_lbnr_set = filter_out_individuals_based_on_basic_df(subpop_basic_df["first_outcome_date"] > subpop_basic_df["dod"], "first outcome diag after death date", sanity_filtered_subpop_lbnr_set)
t("Filtering out individuals with outcome events after dod")


#%%

#t("Filtering out individuals with outcome before subpop event")
#free_of_outcome_at_onset_lbnr_set = filter_out_individuals_based_on_basic_df(subpop_basic_df["first_subpop_event"] < subpop_basic_df["dob"], "first subpop diag before birth date", sanity_filtered_subpop_lbnr_set)
#t("Filtering out individuals with outcome before subpop event")

#%%

t("Followup filtering")
#Filter individuals based on lack of sufficient followup (individual was diagnosed with subpop for the first time too late)

last_calendar_day_of_data = datetime(2016, 1, 1) #|| input

#note to self: python standard datetime lib is retarded
last_valid_calendar_day_of_subpop_diagnosis = datetime(last_calendar_day_of_data.year - followup_period_years, last_calendar_day_of_data.month, last_calendar_day_of_data.day)

followup_filtered_subpop_lbnrs = filter_out_individuals_based_on_basic_df(subpop_basic_df["first_subpop_event"] > last_valid_calendar_day_of_subpop_diagnosis, "too short of a followup", sanity_filtered_subpop_lbnr_set)
#%%

#followup_filtered_subpop_lbnrs = set(followup_filtered_first_subpop_event_df.index)
followup_filtered_subpop_basic_df = subpop_basic_df.loc[subpop_basic_df.index.isin(followup_filtered_subpop_lbnrs)]

followup_filtered_subpop_sks_df = subpop_sks_df.loc[subpop_sks_df["lbnr"].isin(followup_filtered_subpop_lbnrs)]
followup_filtered_subpop_diag_df = subpop_diag_df.loc[subpop_diag_df["lbnr"].isin(followup_filtered_subpop_lbnrs)]
followup_filtered_subpop_prescriptions_df = subpop_prescriptions_df.loc[subpop_prescriptions_df["lbnr"].isin(followup_filtered_subpop_lbnrs)]
followup_filtered_subpop_ssr_df = subpop_ssr_df.loc[subpop_ssr_df["lbnr"].isin(followup_filtered_subpop_lbnrs)]
t("Followup filtering")

#%%

#%%

# Calculate the number of days each individual lived in each region before the t0

def count_days_by_lbnr_and_region(address_df, first_subpop_event_series):
    tmp_address_df = address_df
    tmp_address_df = tmp_address_df.join(first_subpop_event_df, how="left")
    tmp_address_df = tmp_address_df[tmp_address_df["D_TILFLYT_DATO"] < tmp_address_df["first_subpop_event"]]
    if "D_FRAFLYT_DATO" in tmp_address_df.columns:
        tmp_address_df["days_lived_pre_t0"] = np.where(
           (tmp_address_df["D_FRAFLYT_DATO"] > tmp_address_df["first_subpop_event"]) & (tmp_address_df["D_TILFLYT_DATO"] < tmp_address_df["first_subpop_event"]),
            tmp_address_df["first_subpop_event"] - tmp_address_df["D_TILFLYT_DATO"], #If the t0 was in between the move in and move out dates
            tmp_address_df["D_FRAFLYT_DATO"] - tmp_address_df["D_TILFLYT_DATO"] #If the t0 was later than move out date
        )
    else:
        tmp_address_df["days_lived_pre_t0"] = tmp_address_df["first_subpop_event"] - tmp_address_df["D_TILFLYT_DATO"]

    tmp_address_df["days_lived_pre_t0"] = tmp_address_df["days_lived_pre_t0"].dt.days

    #display the calculated columns
    #tmp_address_df[["days_lived_pre_t0", "D_TILFLYT_DATO",  "D_FRAFLYT_DATO", "first_subpop_event"]]

    #display calculated columns for cases where the t0 falls in between the move in and move out dates
    #tmp_address_df[tmp_address_df["D_FRAFLYT_DATO"] >= tmp_address_df["first_subpop_event"]][["days_lived_pre_t0", "D_TILFLYT_DATO",  "D_FRAFLYT_DATO", "first_subpop_event"]]

    lbnr_region_days_lived_until_t0 = tmp_address_df.groupby([tmp_address_df.index, "Region"])["days_lived_pre_t0"].sum()

    return lbnr_region_days_lived_until_t0

t("Assembling address data")
past_archive_addresses_region_days_lived_until_t0 = count_days_by_lbnr_and_region(past_archive_addresses_df, subpop_basic_df["first_subpop_event"])
past_addresses_region_days_lived_until_t0 = count_days_by_lbnr_and_region(past_addresses_df, subpop_basic_df["first_subpop_event"])
current_addresses_region_days_lived_until_t0 = count_days_by_lbnr_and_region(current_addresses_df, subpop_basic_df["first_subpop_event"])


#Sum the days each individual lived in a give region before the t0
subpop_region_days_lived_until_t0_series = current_addresses_region_days_lived_until_t0.add(past_archive_addresses_region_days_lived_until_t0, fill_value=0).add(past_addresses_region_days_lived_until_t0, fill_value=0)

#Arrange the dataframe, assign the int type to values
subpop_region_days_lived_until_t0_df = subpop_region_days_lived_until_t0_series.reset_index(level=["Region"])

subpop_region_days_lived_until_t0_df["days_lived_pre_t0"] = subpop_region_days_lived_until_t0_df["days_lived_pre_t0"].fillna(0).astype(np.int)
t("Assembling address data")

#%%


#%%

from pandas.tseries.offsets import DateOffset

def filter_event_df_by_time_offset(event_df, event_df_time_col_name, time_offset_in_years, remove_type="keep_after", lbnr_col_name = "lbnr", t0_df=followup_filtered_subpop_basic_df):
    event_df_mask = pd.merge(event_df[[lbnr_col_name]], t0_df[["first_subpop_event"]], left_on=lbnr_col_name, right_index=True, how="left")
    if remove_type == "keep_after":
        reduced_event_df = event_df[event_df[event_df_time_col_name] > (event_df_mask["first_subpop_event"] + DateOffset(years=time_offset_in_years))]
    else:
        reduced_event_df = event_df[event_df[event_df_time_col_name] < (event_df_mask["first_subpop_event"] + DateOffset(years=time_offset_in_years))]
    return reduced_event_df

#%%

#This is specified in years as a window around first subpop diagnosis
pre_t0_time_window = -300 #|| input
post_t0_time_window = 0  #|| input



p("Number of observations before time window filtering", {
        "diagnoses": followup_filtered_subpop_diag_df.shape[0],
        "procedures": followup_filtered_subpop_sks_df.shape[0],
        "prescriptions": followup_filtered_subpop_prescriptions_df.shape[0],
        "SSR events": followup_filtered_subpop_ssr_df.shape[0],})

#%%
t("Filtering by requiring events to fall within specific time window around t0")
post_filtered_subpop_diag_df = filter_event_df_by_time_offset(followup_filtered_subpop_diag_df, "D_uddto", post_t0_time_window, "keep_before")
post_filtered_subpop_sks_df = filter_event_df_by_time_offset(followup_filtered_subpop_sks_df, "D_uddto", post_t0_time_window, "keep_before")
post_filtered_subpop_prescriptions_df = filter_event_df_by_time_offset(followup_filtered_subpop_prescriptions_df, "EKSD", post_t0_time_window, "keep_before")
post_filtered_subpop_ssr_df = filter_event_df_by_time_offset(followup_filtered_subpop_ssr_df, "date", post_t0_time_window, "keep_before")

p("Event count reduction based on post-t0 time horizon filtering", {
        "% diagnoses": 100* (followup_filtered_subpop_diag_df.shape[0] - post_filtered_subpop_diag_df.shape[0])/followup_filtered_subpop_diag_df.shape[0],
        "% procedures": 100* (followup_filtered_subpop_sks_df.shape[0] - post_filtered_subpop_sks_df.shape[0])/followup_filtered_subpop_sks_df.shape[0],
        "% prescriptions": 100* (followup_filtered_subpop_prescriptions_df.shape[0] - post_filtered_subpop_prescriptions_df.shape[0])/followup_filtered_subpop_prescriptions_df.shape[0],
        "% SSR events": 100* (followup_filtered_subpop_ssr_df.shape[0] - post_filtered_subpop_ssr_df.shape[0])/followup_filtered_subpop_ssr_df.shape[0],
        })

#%%

if pre_t0_time_window != 0:
    pre_filtered_subpop_diag_df = filter_event_df_by_time_offset(post_filtered_subpop_diag_df, "D_uddto", pre_t0_time_window, "keep_after")
    pre_filtered_subpop_sks_df = filter_event_df_by_time_offset(post_filtered_subpop_sks_df, "D_uddto", pre_t0_time_window, "keep_after")
    pre_filtered_subpop_prescriptions_df = filter_event_df_by_time_offset(post_filtered_subpop_prescriptions_df, "EKSD", pre_t0_time_window, "keep_after")
    pre_filtered_subpop_ssr_df = filter_event_df_by_time_offset(post_filtered_subpop_ssr_df, "date", pre_t0_time_window, "keep_after")
else:
    pre_filtered_subpop_diag_df = post_filtered_subpop_diag_df
    pre_filtered_subpop_sks_df = post_filtered_subpop_sks_df
    pre_filtered_subpop_prescriptions_df = post_filtered_subpop_prescriptions_df
    pre_filtered_subpop_ssr_df = post_filtered_subpop_ssr_df

p("Event count reduction based on pre-t0 time horizon filtering (applied on post-t0 filtered set)", {
        "% diagnoses": 100* (post_filtered_subpop_diag_df.shape[0] - pre_filtered_subpop_diag_df.shape[0])/post_filtered_subpop_diag_df.shape[0],
        "% procedures": 100* (post_filtered_subpop_sks_df.shape[0] - pre_filtered_subpop_sks_df.shape[0])/post_filtered_subpop_sks_df.shape[0],
        "% prescriptions": 100* (post_filtered_subpop_prescriptions_df.shape[0] - pre_filtered_subpop_prescriptions_df.shape[0])/post_filtered_subpop_prescriptions_df.shape[0],
        "% SSR events": 100* (post_filtered_subpop_ssr_df.shape[0] - pre_filtered_subpop_ssr_df.shape[0])/post_filtered_subpop_ssr_df.shape[0],
        })
t("Filtering by requiring events to fall within specific time window around t0")


#%%

#
# Use parent data
#



if use_parent_diagnoses:
    filtered_subpop_parent_diag_df = filter_event_df_by_time_offset(subpop_parents_diag_df, "D_uddto", post_t0_time_window, "keep_before", lbnr_col_name="child_lbnr")
    filtered_subpop_parent_diag_df.loc[:, "lbnr"] = filtered_subpop_parent_diag_df["child_lbnr"]

#%%

filtered_subpop_lbnrs = followup_filtered_subpop_lbnrs

filtered_subpop_basic_df = followup_filtered_subpop_basic_df

filtered_subpop_diag_df = pre_filtered_subpop_diag_df
filtered_subpop_sks_df = pre_filtered_subpop_sks_df
filtered_subpop_prescriptions_df = pre_filtered_subpop_prescriptions_df
filtered_subpop_ssr_df = pre_filtered_subpop_ssr_df

p("Number of observations after time filtering (pre, post and followup requirements)", {
#        "parent_diagnoses": filtered_subpop_parent_diag_df.shape[0] if use_parent_diagnoses else 0,
        "diagnoses": filtered_subpop_diag_df.shape[0],
        "procedures": filtered_subpop_sks_df.shape[0],
        "prescriptions": filtered_subpop_prescriptions_df.shape[0],
        "SSR events": filtered_subpop_ssr_df.shape[0],})

#%%
#Now pivot the dataframes into feature-vectors

code_prefixes = {
    "diag": "DIAG",
    "parent_diag": "PDIAG",
    "atc": "ATC",
    "ssr-speciale": "SSR_SPECIALE",
    "ssr-ydelse": "SSR_YDELSE",
    "sksopr": "SKS_OPR"
}


t("Cut and prefix codes")
def code_cutter(shorten_codes_from, shorten_codes_to):
    def code_cutter_internal(code):
        return code[shorten_codes_from: shorten_codes_to]
    return code_cutter_internal

def code_prefixer(code_type):
    def code_prefixer_internal(code):
        if code_type in code_prefixes:
            return code_prefixes[code_type] + "-" +  str(code)
        else:
            raise Exception("unknown code_type")
    return code_prefixer_internal

def count_variable_occurences_per_person(target_df, event_name_colname):
    mix_df = target_df[["lbnr", event_name_colname]]
    mix_df.loc[:, "count"] = 1
    lbnr_event_count_df = mix_df.groupby(["lbnr", event_name_colname]).count()#lbnr can also be mix_df.index
    return lbnr_event_count_df

def shorten_and_prefix_codes(target_df, target_df_valuecol, code_cutter_func, code_prefixer_func):
    if code_cutter_func:
        target_df.loc[:, target_df_valuecol] = target_df[target_df_valuecol].apply(code_cutter_func)
    if code_prefixer_func:
        target_df.loc[:, target_df_valuecol] = target_df[target_df_valuecol].apply(code_prefixer_func)
    return target_df

def filter_event_dataframes_by_feature_support(event_df, event_col_name, min_req_support):
    unique_lbnrevent_pairs_df = event_df.loc[:, ["lbnr", event_col_name]].drop_duplicates(subset=None, keep="first", inplace=False)
    event_counts_series = unique_lbnrevent_pairs_df[event_col_name].value_counts(normalize=False)
    support_filtered_events = set(event_counts_series[event_counts_series > min_req_support].index)

    suport_filtered_event_df = event_df[event_df[event_col_name].isin(support_filtered_events)]

    return suport_filtered_event_df

def prefix_shorten_filter_count_pivot_events_df(df, df_valuecol, code_cutter_func, code_prefixer_func, min_req_support):
    target_df = df.copy()
    target_df = shorten_and_prefix_codes(target_df, df_valuecol, code_cutter_func, code_prefixer_func)
    #target_df = target_df.drop_duplicates() #Remove duplicates in case shortened codes lead to duplication
    target_df = filter_event_dataframes_by_feature_support(target_df, df_valuecol, min_req_support)
    target_df = count_variable_occurences_per_person(target_df, df_valuecol)
    print(f"prefix shorten filter count pivot target_df_shape: {target_df.shape}")
    pivoted_target_df = target_df.unstack(level=-1,fill_value=0)
    #The columns will now have two level index as "count" column becomes one
    pivoted_target_df.columns = pivoted_target_df.columns.droplevel(0)
    return pivoted_target_df


def convert_to_one_hot(df):
    one_hot_df = df.copy()
    one_hot_df[df > 0] = 1
    one_hot_df.fillna(0)
    return one_hot_df


#def filter_features_based_on_min_required_support(event_df, min_spport):
#    one_hot_df = convert_to_one_hot(event_df)
#    filtered_columns = one_hot_df.loc[:, one_hot_df.sum(axis=0) > diag_feature_min_support].columns
#    return event_df.loc[:, filtered_columns]

#%%
#this is a test for count_prefix_shorten_pivot_events_df
#df = filtered_subpop_diag_df
#df_valuecol = "C_diag"
#code_cutter_func = code_cutter(0, 3)
#code_prefixer_func = code_prefixer("diag")
#%%

# Calculate code reduction/support matrix
# Note the below doesn't work as it crashes with ValueError: negative dimensions are not allowed
# It seems as if the diemnsions are simply too big to unstack - a sparse matrix might be a good solution
#code_length_support_df = pd.DataFrame()
#for max_code_length in [3, 4]:
#    reduced_features_df = count_prefix_shorten_pivot_events_df(filtered_subpop_diag_df, "C_diag", code_cutter(0, max_code_length), None)
#    for min_support in [0, 20, 40, 100, 200]:
#        code_length_support_df.loc[min_support, max_code_length] = filter_features_based_on_min_required_support(reduced_features_df, min_support).shape[1]
#%%
# Here we convert event dataframes to event count (feature) dataframes

use_diag_codes_up_to_length = 4 #|| input
use_proc_codes_up_to_length = 5 #|| input
use_drug_codes_up_to_length = 5 #|| input

diag_feature_min_support = 50 #|| input
sksopr_feature_min_support = 50 #|| input
prescription_feature_min_support = 50 #|| input
outpatient_visit_feature_min_support = 50 #|| input

print("diagnoses")
diag_features_df = pd.concat([
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_diag_df, "C_diag", code_cutter(0, 1), code_prefixer("diag"), diag_feature_min_support) if use_diag_codes_up_to_length >= 1 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_diag_df, "C_diag", code_cutter(0, 3), code_prefixer("diag"), diag_feature_min_support) if use_diag_codes_up_to_length >= 3 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_diag_df, "C_diag", code_cutter(0, 4), code_prefixer("diag"), diag_feature_min_support) if use_diag_codes_up_to_length >= 4 else None], axis=1)
diag_features_df = diag_features_df.loc[:, ~diag_features_df.columns.duplicated(keep="first")] #Concatenating of above may lead to duplicate columns, only the first one is kept


#diag_features_df = count_prefix_shorten_pivot_events_df(filtered_subpop_diag_df, "C_diag", code_cutter(0, 3), code_prefixer("diag"))
print("sksopr")
sksopr_features_df = pd.concat([
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_sks_df, "C_opr", code_cutter(0, 1), code_prefixer("sksopr"), sksopr_feature_min_support) if use_proc_codes_up_to_length >= 1 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_sks_df, "C_opr", code_cutter(0, 2), code_prefixer("sksopr"), sksopr_feature_min_support) if use_proc_codes_up_to_length >= 2 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_sks_df, "C_opr", code_cutter(0, 3), code_prefixer("sksopr"), sksopr_feature_min_support) if use_proc_codes_up_to_length >= 3 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_sks_df, "C_opr", code_cutter(0, 5), code_prefixer("sksopr"), sksopr_feature_min_support) if use_proc_codes_up_to_length >= 5 else None], axis=1)
sksopr_features_df = sksopr_features_df.loc[:, ~sksopr_features_df.columns.duplicated(keep="first")] #Concatenating of above may lead to duplicate columns, only the first one is kept
#sksopr_features_df = count_prefix_shorten_pivot_events_df(filtered_subpop_sks_df, "C_opr", code_cutter(0, 4), code_prefixer("sksopr"))

print("medications")
prescription_features_df = pd.concat([
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_prescriptions_df, "ATC", code_cutter(0, 1), code_prefixer("atc"), prescription_feature_min_support) if use_drug_codes_up_to_length >= 1 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_prescriptions_df, "ATC", code_cutter(0, 3), code_prefixer("atc"), prescription_feature_min_support) if use_drug_codes_up_to_length >= 3 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_prescriptions_df, "ATC", code_cutter(0, 4), code_prefixer("atc"), prescription_feature_min_support) if use_drug_codes_up_to_length >= 4 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_prescriptions_df, "ATC", code_cutter(0, 5), code_prefixer("atc"), prescription_feature_min_support) if use_drug_codes_up_to_length >= 5 else None,
        prefix_shorten_filter_count_pivot_events_df(filtered_subpop_prescriptions_df, "ATC", code_cutter(0, 7), code_prefixer("atc"), prescription_feature_min_support) if use_drug_codes_up_to_length >= 7 else None], axis=1)
prescription_features_df = prescription_features_df.loc[:, ~prescription_features_df.columns.duplicated(keep="first")] #Concatenating of above may lead to duplicate columns, only the first one is kept

if use_parent_diagnoses:
    print("parent_diagnoses")
    parent_diag_features_df = prefix_shorten_filter_count_pivot_events_df(filtered_subpop_parent_diag_df, "C_diag", code_cutter(0, 3), code_prefixer("parent_diag"), diag_feature_min_support) if use_parent_diagnoses else None
else:
    parent_diag_features_df = pd.DataFrame()
#prescription_features_df = count_prefix_shorten_pivot_events_df(filtered_subpop_prescriptions_df, "ATC", code_cutter(0, 4), code_prefixer("atc"))

#SSR contains more detailed information than just the type of medical professional that was contacted/visited, this likely leads to duplicates if we just use C_SPECIALE column which are stripped here
reduced_filtered_subpop_ssr_df = filtered_subpop_ssr_df.loc[:, ["lbnr", "C_SPECIALE"]].drop_duplicates()

outpatient_visit_features_df = prefix_shorten_filter_count_pivot_events_df(reduced_filtered_subpop_ssr_df, "C_SPECIALE", code_cutter(0, 100), code_prefixer("ssr-speciale"), outpatient_visit_feature_min_support)

address_features_df = subpop_region_days_lived_until_t0_df.pivot(index=None, columns="Region").fillna(0).astype(np.int) #Pivot Regions to become columns (features), fill NAs with zeros, change type again as it's converted to float (why?!?)
address_features_df.columns = ["ADDR " + region for region in address_features_df.columns.droplevel(0)]

t("Cut and prefix codes")

#%%





#%%

# Here we filter feautures within event count (feature) dataframes based on minimum required support for each feature
# This is outdated as support filtering was built into the pivot procedure. It was necessary to reduce sparsity early and prevent pandas from crashing

#def get_prcnt_size(after_df, before_df, dimension=0):
#    return 100 - 100*(before_df.shape[dimension] - after_df.shape[dimension])/before_df.shape[dimension]
#
#t("Filter features by support")
#diag_feature_min_support = 50 #|| input
#sksopr_feature_min_support = 50 #|| input
#prescription_feature_min_support = 50 #|| input
#outpatient_visit_feature_min_support = 50 #|| input
#
#filtered_diag_features_df = filter_features_based_on_min_required_support(diag_features_df, diag_feature_min_support)
#filtered_sksopr_features_df = filter_features_based_on_min_required_support(sksopr_features_df, sksopr_feature_min_support)
#filtered_prescription_features_df = filter_features_based_on_min_required_support(prescription_features_df, prescription_feature_min_support)
#filtered_outpatient_visit_features_df = filter_features_based_on_min_required_support(outpatient_visit_features_df, outpatient_visit_feature_min_support)
#
#p("Number of features removed due to required feature support", {
#    "diagnoses": {
#            "required_feature_support": diag_feature_min_support,
#            "# features after filtering": filtered_diag_features_df.shape[1],
#            "% features after filtering": get_prcnt_size(filtered_diag_features_df, diag_features_df, 1)
#    },
#    "procedures": {
#            "required_feature_support": sksopr_feature_min_support,
#            "# features after filtering": filtered_sksopr_features_df.shape[1],
#            "% features after filtering": get_prcnt_size(filtered_sksopr_features_df, sksopr_features_df, 1)
#    },
#    "prescriptions": {
#            "required_feature_support": prescription_feature_min_support,
#            "# features after filtering": filtered_prescription_features_df.shape[1],
#            "% features after filtering": get_prcnt_size(filtered_prescription_features_df, prescription_features_df, 1)
#    },
#    "oupatient_visits": {
#            "required_feature_support": outpatient_visit_feature_min_support,
#            "# features after filtering": filtered_outpatient_visit_features_df.shape[1],
#            "% features after filtering": get_prcnt_size(filtered_outpatient_visit_features_df, outpatient_visit_features_df, 1)
#    }
#})
#t("Filter features by support")

#%%

# filtered_subpop_basic_df = subpop_basic_df

# Create the baseline features data frame
t("Assemble features for baseline model (\"Aux features\"")
#year_to_count_as_zero = min(filtered_subpop_basic_df["dob"].dt.year)
#if any(filtered_subpop_basic_df["dob"].dt.year < year_to_count_as_zero):
#    raise Exception(f"Someone was born before {year_to_count_as_zero} and the below will fail")

basic_features_df = pd.DataFrame()

#basic_features_df["dob_m"] = ((filtered_subpop_basic_df["dob"].dt.year - year_to_count_as_zero)* 12) + filtered_subpop_basic_df["dob"].dt.month
basic_features_df["dob"] = (filtered_subpop_basic_df["dob"] - min(filtered_subpop_basic_df["dob"])).dt.days
#basic_features_df["time_of_first_subpop_m"] = ((filtered_subpop_basic_df["first_subpop_event"].dt.year - year_to_count_as_zero)* 12) + filtered_subpop_basic_df["first_subpop_event"].dt.month
basic_features_df["time_of_first_subpop"] = (filtered_subpop_basic_df["first_subpop_event"] - min(filtered_subpop_basic_df["first_subpop_event"])).dt.days
age_at_first_subpop = filtered_subpop_basic_df["first_subpop_event"] - filtered_subpop_basic_df["dob"]
basic_features_df["age_at_first_subpop"] = (age_at_first_subpop - min(age_at_first_subpop)).dt.days
basic_features_df["gender"] = ["Male" if val == 1 else "Female" for val in filtered_subpop_basic_df["gender"]]
basic_features_df["pob"] = filtered_subpop_basic_df["fodested"].fillna("Unknown")

#pob_category_index_dict = {cat_name : cat_index for cat_index, cat_name in enumerate(basic_features_df["pob"].cat.categories)}
#basic_features_df["pob"] = basic_features_df["pob"].cat.rename_categories(range(len(pob_category_index_dict)))
#basic_features_df["pob"] = basic_features_df["pob"].astype(np.int)


p("Basic features fomulation", {
        "earliest dob": str(min(filtered_subpop_basic_df["dob"])),
        "earliest date of first subpop": str(min(filtered_subpop_basic_df["first_subpop_event"]))
        })



#Patsy code that works:


#import numpy as np
#
extended_basic_features_df = dmatrix("time_of_first_subpop + dob + C(gender) + C(pob) + age_at_first_subpop:C(gender) + bs(time_of_first_subpop, df=4, degree=3, include_intercept=False)  + bs(age_at_first_subpop, df=4, degree=3, include_intercept=False) - 1",
                                          data = basic_features_df,
                                          return_type = "dataframe")




#Patsy bug to report

#from patsy import dmatrix
#import numpy as np
#data1 = {"gender": [0,1,0,1,0,0]}
#data2 = np.rec.fromrecords([(0,),(1,),(0,),(1,),(0,),(0,)], names="gender")
#dmatrix("C(gender)", data1) #works
#dmatrix("C(gender)", data2) #broken

#####

t("Assemble features for baseline model (\"Aux features\"")

#p("Place of birth categories map", pob_category_index_dict)

#%%

# Divide cases and controls
t("Divide cases and controls")
case_lbnr_set = set(filtered_subpop_basic_df[(filtered_subpop_basic_df["first_outcome_date"].notnull()) &
                                           (filtered_subpop_basic_df["first_outcome_date"] > (filtered_subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=buffer_period_length))) &
                                           (filtered_subpop_basic_df["first_outcome_date"] < (filtered_subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=365*followup_period_years + buffer_period_length)))].index)

control_lbnr_set = set(filtered_subpop_basic_df[(filtered_subpop_basic_df["first_outcome_date"].isnull()) |
                                             (filtered_subpop_basic_df["first_outcome_date"] > (filtered_subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=365*followup_period_years + buffer_period_length)))].index)


assert(len(case_lbnr_set & control_lbnr_set) == 0)

p("Case and control division", {
        "# cases": len(case_lbnr_set),
        "% cases": 100.0 * len(case_lbnr_set)/(len(case_lbnr_set) + len(control_lbnr_set)),
        "# cases identified through death register": len(case_lbnrs_identified_through_deathreg & case_lbnr_set),
        "% cases identified through death register": 100.0 * len(case_lbnrs_identified_through_deathreg & case_lbnr_set) / len(case_lbnr_set),
        "# controls": len(control_lbnr_set),
        "% controls": 100.0 * len(control_lbnr_set)/(len(case_lbnr_set) + len(control_lbnr_set)),
        f"# individuals with outcome before or within {buffer_period_length} days of outcome": filtered_subpop_basic_df[(filtered_subpop_basic_df["first_outcome_date"].notnull()) &
                                                                          (filtered_subpop_basic_df["first_outcome_date"] < (filtered_subpop_basic_df["first_subpop_event"] + pd.Timedelta(days=buffer_period_length)))].shape[0]})

case_lbnr_set #|| output
control_lbnr_set #|| output
t("Divide cases and controls")

#%%

# Update filtered_subpop_basic_df to reflect only cases and controls (it's just used to show final summary stats after filtering)
filtered_subpop_basic_df = filtered_subpop_basic_df[filtered_subpop_basic_df.index.isin(case_lbnr_set | control_lbnr_set)]

#%%

t("Assemble combined feature dataframe")
features_dataframe_list = [extended_basic_features_df, address_features_df, diag_features_df, sksopr_features_df, prescription_features_df, outpatient_visit_features_df]
if use_parent_diagnoses:
    features_dataframe_list.append(parent_diag_features_df)

feature_df = pd.concat(features_dataframe_list, axis=1)
feature_df = feature_df.fillna(value=0)
feature_df = feature_df.astype(np.int)
feature_df = feature_df[feature_df.index.isin(case_lbnr_set | control_lbnr_set)] #Take only cases or controls
t("Assemble combined feature dataframe")

#%%

#t_feature_df = feature_df.transpose()
#%%
#t_feature_df["groups"] = t_feature_df.index
#t_feature_df["groups"] = t_feature_df["groups"].str.s

#%%


#Calculate feature statistics
#stat_diag_features_df = prefix_shorten_filter_count_pivot_events_df(filtered_subpop_diag_df, "C_diag", code_cutter(0, 1), code_prefixer("diag"), diag_feature_min_support)
#stat_opr_features_df = prefix_shorten_filter_count_pivot_events_df(filtered_subpop_sks_df, "C_opr", code_cutter(0, 1), code_prefixer("sksopr"), sksopr_feature_min_support)
#stat_presc_features_df = prefix_shorten_filter_count_pivot_events_df(filtered_subpop_prescriptions_df, "ATC", code_cutter(0, 1), code_prefixer("atc"), prescription_feature_min_support)
#outpatient_visit_features_df = prefix_shorten_filter_count_pivot_events_df(filtered_subpop_ssr_df, "C_SPECIALE", code_cutter(0, 100), code_prefixer("ssr-speciale"), outpatient_visit_feature_min_support)


#
#Now the trick here is that  due to keeping hierarchical information in the input, single event would normally be counted as many
#A soultion to this is to take only the shortest code
#
#Feature counts answers: On average an individual will have X amount of diagnoses/medications etc
#Feature support answers: On averge a feature will have X amount of individuals with non-zero value
def get_summary_statistics_for_feature_support(feature_df):
    t_feature_df = feature_df.transpose()
    #if summary_over_popoulation == False:
    #    t_feature_df = t_feature_df.sum(axis=1)
    #feature_prefixes_to_prune = {"DIAG", "ATC", "SKS_OPR"}
    t_feature_df["feature_group"] = t_feature_df.index.str.split("-").str.get(0)
    t_feature_reduced_df = t_feature_df[t_feature_df["feature_group"].isin(code_prefixes.values())]
    feature_grouping = t_feature_reduced_df.groupby(by=["feature_group"])
    #aggregated_feature_means_per_person = feature_grouping.mean() #DO not use these, think
    aggregated_feature_sums_per_person = feature_grouping.sum() #
    feature_counts = feature_grouping.size()

    feature_type_support_in_population = {}
    for code_type, code_feature_prefix in code_prefixes.items():
        if code_feature_prefix in aggregated_feature_sums_per_person.index:
            feature_type_support_in_population[code_type] = my_describe(aggregated_feature_sums_per_person.loc[code_feature_prefix])
            feature_type_support_in_population[code_type]["count_features"] = feature_counts[code_feature_prefix]

    return feature_type_support_in_population

p("Feature type counts for each individual", get_summary_statistics_for_feature_support(feature_df))
#p("Feature counts for each feature type", get_summary_statistics_for_feature_support(feature_df, summary_over_popoulation=False))






#%%
subpop_basic_df #|| output

basic_features_df #|| output
extended_basic_features_df #|| output
diag_features_df #|| output
parent_diag_features_df #|| output
sksopr_features_df #|| output
prescription_features_df #|| output
outpatient_visit_features_df #|| output

address_features_df #|| output

feature_df #|| output

patient_feature_vector_arr = feature_df.values #|| output

#%%












































#%%

num_aux_features = extended_basic_features_df.shape[1] #|| output

patient_feature_vector___index__to__feature__list = list(feature_df.columns) #|| output
aux__patient_feature_vector___index__to__feature__list = patient_feature_vector___index__to__feature__list[0:num_aux_features] #|| output


patient_feature_vector___feature__to__index__map = {feature: index for index, feature in enumerate(patient_feature_vector___index__to__feature__list)} #|| output

patient_feature_vector___index__to__lbnr__list = list(feature_df.index) #|| output
patient_feature_vector___lbnr__to__index__map = {lbnr: index for index, lbnr in enumerate(patient_feature_vector___index__to__lbnr__list)} #|| output

#NOTE below is useful for testing
#test_first_subpop_event = subpop_basic_df.ix[239]["first_subpop_event"]
#filtered_subpop_diag_df[(filtered_subpop_diag_df["lbnr"] == 239) & (filtered_subpop_diag_df['D_inddto'] < test_first_subpop_event)]['C_diag'].apply(lambda x: x[:4]).value_counts().to_dict()
#row_i = patient_feature_vector___lbnr__to__index__map[239]
#col_i = patient_feature_vector___feature__to__index__map["DIAG-N81"]
#patient_feature_vector_arr[row_i, col_i]

#%%

p("Post-filtering", {
        "# individuals": filtered_subpop_basic_df.shape[0],
        "# individuals with outcome": filtered_subpop_basic_df[filtered_subpop_basic_df["first_outcome_date"].notnull()].shape[0],
        "year of birth description": my_describe(filtered_subpop_basic_df["yob"]),
        "% women": 100.0 * sum(filtered_subpop_basic_df["gender"] == 0) / filtered_subpop_basic_df["gender"].shape[0],
        "age of first subpop event description": my_describe((filtered_subpop_basic_df["first_subpop_event"] - filtered_subpop_basic_df["dob"]).dt.days/365),
        "year of first subpop event description": my_describe(filtered_subpop_basic_df["first_subpop_event"].dt.year),
        "time difference between subpop event and outcome event description": my_describe((filtered_subpop_basic_df["first_outcome_date"] - filtered_subpop_basic_df["first_subpop_event"]).dt.days),
        "time difference between subpop event and death event description": my_describe((filtered_subpop_basic_df["dod"] - filtered_subpop_basic_df["first_subpop_event"]).dt.days)
})

p("Post-filtering case description", {
    "# individuals": filtered_subpop_basic_df.loc[case_lbnr_set].shape[0],
    "year of birth description": my_describe(filtered_subpop_basic_df.loc[case_lbnr_set]["yob"]),
    "% women": 100.0 * sum(filtered_subpop_basic_df.loc[case_lbnr_set]["gender"] == 0) / filtered_subpop_basic_df.loc[case_lbnr_set]["gender"].shape[0],
    "age of first subpop event description": my_describe((filtered_subpop_basic_df.loc[case_lbnr_set]["first_subpop_event"] - filtered_subpop_basic_df.loc[case_lbnr_set]["dob"]).dt.days/365),
    "year of first subpop event description": my_describe(filtered_subpop_basic_df.loc[case_lbnr_set]["first_subpop_event"].dt.year),
    "time difference between subpop event and outcome event description": my_describe((filtered_subpop_basic_df.loc[case_lbnr_set]["first_outcome_date"] - filtered_subpop_basic_df.loc[case_lbnr_set]["first_subpop_event"]).dt.days),
    "time difference between subpop event and death event description": my_describe((filtered_subpop_basic_df.loc[case_lbnr_set]["dod"] - filtered_subpop_basic_df.loc[case_lbnr_set]["first_subpop_event"]).dt.days)
})

p("Post-filtering controls description", {
    "# individuals": filtered_subpop_basic_df.loc[control_lbnr_set].shape[0],
    "year of birth description": my_describe(filtered_subpop_basic_df.loc[control_lbnr_set]["yob"]),
    "% women": 100.0 * sum(filtered_subpop_basic_df.loc[control_lbnr_set]["gender"] == 0) / filtered_subpop_basic_df.loc[control_lbnr_set]["gender"].shape[0],
    "age of first subpop event description": my_describe((filtered_subpop_basic_df.loc[control_lbnr_set]["first_subpop_event"] - filtered_subpop_basic_df.loc[control_lbnr_set]["dob"]).dt.days/365),
    "year of first subpop event description": my_describe(filtered_subpop_basic_df.loc[control_lbnr_set]["first_subpop_event"].dt.year),
    "time difference between subpop event and outcome event description": my_describe((filtered_subpop_basic_df.loc[control_lbnr_set]["first_outcome_date"] - filtered_subpop_basic_df.loc[control_lbnr_set]["first_subpop_event"]).dt.days),
    "time difference between subpop event and death event description": my_describe((filtered_subpop_basic_df.loc[control_lbnr_set]["dod"] - filtered_subpop_basic_df.loc[control_lbnr_set]["first_subpop_event"]).dt.days)
})


#%%





##%%
##TODO just a marker
##||
#action_name = "Calculate stats about feature vector"
#action_description = "Calculate stats about feature vector"
#action_output = {"subpop_lbnrs", "first_subpop_event_df"}
##||
#
#cpr_df = load_input(load_input_run_name, "cpr_df", load_input_shared_storage_name) #|| input
#Xth_birthday = 65 #|| input
#Xth_birthday_min_date = date(2000, 1, 1) #|| input
#Xth_birthday_max_date = date(2016, 1, 1) #|| input
#p("inputs", {
#        "Xth_birthday": Xth_birthday,
#        "Xth_birthday_min_date": Xth_birthday_min_date,
#        "Xth_birthday_max_date": Xth_birthday_max_date})
##we don't strictly need to correct for max date as followup filtering will adjust this but this is measure of decreasing the dataset size
#
#Xth_bd__intermediate_df = pd.DataFrame(
#        {"year": cpr_df["D_foddato"].dt.year + Xth_birthday,
#         "month": cpr_df["D_foddato"].dt.month,
#         "day": cpr_df["D_foddato"].dt.day})
#
#if Xth_birthday % 4: #If it's not a leap year -then fix leaper birth dates
#    leapers = (Xth_bd__intermediate_df["month"] == 2) & (Xth_bd__intermediate_df["day"] == 29)
#    Xth_bd__intermediate_df.loc[leapers, "month"] = 3
#    Xth_bd__intermediate_df.loc[leapers, "day"] = 1
#
#
#cpr_df["Xth_bd"] = pd.to_datetime(Xth_bd__intermediate_df)
#
#time_filtered_cpr_df = cpr_df[cpr_df["Xth_bd"] > Xth_birthday_min_date]
#time_filtered_cpr_df = time_filtered_cpr_df[time_filtered_cpr_df["Xth_bd"] < Xth_birthday_max_date]
#time_filtered_cpr_df = time_filtered_cpr_df[(time_filtered_cpr_df["Xth_bd"] < time_filtered_cpr_df["dod"]) | time_filtered_cpr_df["dod"].isnull()]
#
#
#first_subpop_event_df = pd.DataFrame()
#first_subpop_event_df["first_subpop_event"] = time_filtered_cpr_df["Xth_bd"]
#subpop_lbnrs = set(time_filtered_cpr_df["Xth_bd"].index)

#%%






#%%
#TODO just a marker
#||
action_name = "Divide feature set into training, test and validation. Generate K-fold indexes."
action_description = ""
action_input = {
    "subpop_basic_df",
    "case_lbnr_set",
    "patient_feature_vector_arr",
    "patient_feature_vector___lbnr__to__index__map",
    "patient_feature_vector___index__to__lbnr__list"
}
#||

from sklearn.cross_validation import train_test_split


use_time_based_split = False #|| input

num_aux_features = 4 #||input

num_k_folds = 3 #|| input
training_set_ratio = 0.7 #|| input
test_set_ratio = 0.2 #|| input
val_set_ratio = 1.0 - test_set_ratio - training_set_ratio

testval_set_ratio = 1.0-training_set_ratio #needed for first split
testval_test_set_ratio = test_set_ratio / (val_set_ratio + test_set_ratio)
testval_val_set_ratio = val_set_ratio / (val_set_ratio + test_set_ratio)

subpop_basic_df = load_input(load_input_run_name, "subpop_basic_df", load_input_shared_storage_name) #|| input
case_lbnr_set = load_input(load_input_run_name, "case_lbnr_set", load_input_shared_storage_name) #|| input
patient_feature_vector_arr = load_input(load_input_run_name, "patient_feature_vector_arr", load_input_shared_storage_name) #|| input
patient_feature_vector___lbnr__to__index__map = load_input(load_input_run_name, "patient_feature_vector___lbnr__to__index__map", load_input_shared_storage_name) #|| input
patient_feature_vector___index__to__lbnr__list = load_input(load_input_run_name, "patient_feature_vector___index__to__lbnr__list", load_input_shared_storage_name) #|| input

#%%
feature_df = load_input(load_input_run_name, "feature_df", load_input_shared_storage_name) #|| input
#%%

X = patient_feature_vector_arr

Y = np.zeros(patient_feature_vector_arr.shape[0], dtype=np.int8)
for case_lbnr in case_lbnr_set:
    Y[patient_feature_vector___lbnr__to__index__map[case_lbnr]] = 1

#%%

#%% Time based

if use_time_based_split:
    pfv__index_to_lbnr_df = pd.DataFrame(patient_feature_vector___index__to__lbnr__list).rename(columns={0: "lbnr"})

    merged_pfv__index_to_lbnr_df = pd.merge(pfv__index_to_lbnr_df.reset_index().rename(columns={"index": "pfv_index"}), subpop_basic_df[["first_subpop_event"]], left_on="lbnr", right_index=True, how="left")
    sorted_merged_pfv__index_to_lbnr_df = merged_pfv__index_to_lbnr_df.sort_values(by="first_subpop_event")

    training_set_max_sorted_index = int(training_set_ratio * sorted_merged_pfv__index_to_lbnr_df.shape[0])

    #Training
    train__sorted_merged_pfv__index_to_lbnr_df = sorted_merged_pfv__index_to_lbnr_df.iloc[:training_set_max_sorted_index]
    train__shuffled_merged_pfv__index_to_lbnr_df = shuffle(train__sorted_merged_pfv__index_to_lbnr_df)
    trainX_indexes = list(train__shuffled_merged_pfv__index_to_lbnr_df["pfv_index"])

    #Test & Validation
    testval__sorted_merged_pfv__index_to_lbnr_df = sorted_merged_pfv__index_to_lbnr_df.iloc[training_set_max_sorted_index:]
    testval__shuffled_merged_pfv__index_to_lbnr_df = shuffle(testval__sorted_merged_pfv__index_to_lbnr_df)
    test_set_max_sorted_index = int(testval_test_set_ratio * testval__shuffled_merged_pfv__index_to_lbnr_df.shape[0])
    testX_indexes = list(testval__shuffled_merged_pfv__index_to_lbnr_df.iloc[:test_set_max_sorted_index]["pfv_index"])
    valX_indexes = list(testval__shuffled_merged_pfv__index_to_lbnr_df.iloc[test_set_max_sorted_index:]["pfv_index"])
else: #standard
    X_indexes = np.arange(X.shape[0])
    trainX_indexes, tempX_indexes, trainY, tempY = train_test_split(X_indexes, Y, test_size=testval_set_ratio, random_state=0, stratify=Y)
    testX_indexes, valX_indexes, testY, valY = train_test_split(tempX_indexes, tempY, test_size=testval_val_set_ratio, random_state=0, stratify=tempY)
#%%

assert(len(set(trainX_indexes) & set(testX_indexes)) == 0)
assert(len(set(valX_indexes) & set(testX_indexes)) == 0)
assert(len(set(valX_indexes) & set(trainX_indexes)) == 0)
assert(len(trainX_indexes) > 2*len(testX_indexes))
assert(len(trainX_indexes) > 2*len(valX_indexes))
assert(len(testX_indexes) >= len(valX_indexes))

#%%

trainY = Y[trainX_indexes]
testY = Y[testX_indexes]
valY = Y[valX_indexes]

count__trainX = X[trainX_indexes, :]
count__testX = X[testX_indexes, :]
count__valX = X[valX_indexes, :]

trainX_index__to__patient_feature_vector_indexes = trainX_indexes #|| output
testX_index__to__patient_feature_vector_indexes = testX_indexes #|| output
valX_index__to__patient_feature_vector_indexes = valX_indexes #|| output


p("training set shape", count__trainX.shape)
p("test set shape", count__testX.shape)
p("validation set shape", count__valX.shape)

p("prop cases in training set", sum(trainY)/count__trainX.shape[0])
p("test set shape", sum(testY)/count__testX.shape[0])
p("validation set shape", sum(valY)/count__valX.shape[0])


#%% Time based

#p("last training set onset date", train__sorted_merged_pfv__index_to_lbnr_df.iloc[-1]["first_subpop_event"])


#for i in choice(count__trainX.shape[0], 10):
#    X_i = trainX_indexes[i]
#    i_Xvals = count__trainX[i]
#    F_i = feature_df.iloc[X_i,:].values
#    assert((i_Xvals == F_i).all())
#

#for i in choice(len(testX_indexes), 10):
#    pfv_i = testX_index__to__patient_feature_vector_indexes[i]
#    lbnr_i = patient_feature_vector___index__to__lbnr__list[pfv_i]
#    pred_lbnr = merged_pfv__index_to_lbnr_df[merged_pfv__index_to_lbnr_df["pfv_index"] == pfv_i]["lbnr"]
#    assert(pred_lbnr.shape[0] == 1)
#    assert(pred_lbnr.iloc[0] == lbnr_i)


#%%
# Survival day diff data
#

patient_feature_vector___index__to__lbnr__arr = np.array(patient_feature_vector___index__to__lbnr__list)
subpop_event__to__outcome__daydiff = (subpop_basic_df["first_outcome_date"] - subpop_basic_df["first_subpop_event"]).dt.days
daydiff__trainY = subpop_event__to__outcome__daydiff.loc[patient_feature_vector___index__to__lbnr__arr[trainX_index__to__patient_feature_vector_indexes]]
daydiff__testY = subpop_event__to__outcome__daydiff.loc[patient_feature_vector___index__to__lbnr__arr[testX_index__to__patient_feature_vector_indexes]]
daydiff__valY = subpop_event__to__outcome__daydiff.loc[patient_feature_vector___index__to__lbnr__arr[valX_index__to__patient_feature_vector_indexes]]

#%%
#Multi-hot representation
def binarize(nonbinary_arr):
    binary_arr = np.where(nonbinary_arr == 0, 0, 1)
    binary_arr[:, 0:num_aux_features] = nonbinary_arr[:, 0:num_aux_features] #To accomodate for the first three columns being birth date, gender and t0 rather than counts
    return binary_arr

multihot__trainX = binarize(count__trainX)
multihot__testX = binarize(count__testX)
multihot__valX = binarize(count__valX)

aux__trainX = multihot__trainX[:, 0:num_aux_features]
aux__testX = multihot__testX[:, 0:num_aux_features]
aux__valX = multihot__valX[:, 0:num_aux_features]

n_samples = count__trainX.shape[0]
fold_indexes = KFold(n=n_samples, n_folds=num_k_folds, shuffle=True, random_state=0)


#%%
#Balanced trainig set

from sklearn.utils import shuffle

# Note that we only do that for training set. Test set should have real distribution (and match validation set)

def balance_0_1_classes(X, Y):
    unq, unq_idx, unq_cnt = np.unique(Y, return_inverse=True, return_counts=True)
    num_controls = unq_cnt[0]

    balanced_X = np.empty((num_controls*2, X.shape[1]),  X.dtype)
    balanced_Y = np.zeros((num_controls*2, 1), X.dtype)

    #original
    #slices = np.concatenate(([0], np.cumsum(cnt - unq_cnt)))
    #for j in range(len(unq)):
    #    indices = np.random.choice(np.where(unq_idx==j)[0], cnt - unq_cnt[j])
    #    balanced_X[slices[j]:slices[j+1]] = trainX[indices]

    #fill with class 0 (so that we get all of the controls)
    control_indices = np.where(unq_idx==0)[0]
    balanced_X[0:len(control_indices)] = X[control_indices]

    #fill with class 1 (we oversample our cases)
    case_indices = np.random.choice(np.where(unq_idx==1)[0], num_controls)
    balanced_X[len(control_indices):] = X[case_indices]
    balanced_Y[len(control_indices):] = 1

    #shuffle the two arrays
    balanced_X, balanced_Y = shuffle(balanced_X, balanced_Y, random_state=0)

    #flatten Y array
    balanced_Y = balanced_Y.flatten()

    return balanced_X, balanced_Y

balanced_count__trainX, balanced_count__trainY = balance_0_1_classes(count__trainX, trainY) #note that trainY will be shuffled and extended inside the above procedure
balanced_count__fold_indexes = KFold(n=balanced_count__trainX.shape[0], n_folds=num_k_folds, shuffle=True, random_state=0)

#%%
#Old way of doing split - for some reason yielding weird sizes
#Porbably i mess up usage of indexes
#from sklearn.cross_validation import StratifiedShuffleSplit
#
#train_index, temp_index = next(iter(StratifiedShuffleSplit(Y, n_iter=1, train_size=0.7, random_state=0)))
#test_index, val_index = next(iter(StratifiedShuffleSplit(Y[temp_index], n_iter=1, train_size=0.66, random_state=1)))
#
#trainX, trainY = X[train_index], Y[train_index]
#testX, testY = X[temp_index][test_index], Y[temp_index][test_index]
#valX, valY = X[temp_index][val_index], Y[temp_index][val_index]
#
#print("training set shape: {}".format(trainX.shape))
#print("test set shape: {}".format(testX.shape))
#print("validation set shape: {}".format(valX.shape))
#%%

#%%
print("Save dataset to disk")

fold_indexes #|| output

count__trainX #|| output
count__testX #|| output
count__valX #|| output

trainY #|| output
testY #|| output
valY #|| output

multihot__trainX #|| output
multihot__testX #|| output
multihot__valX #|| output

aux__trainX #|| output
aux__testX #|| output
aux__valX #|| output

balanced_count__fold_indexes #|| output
balanced_count__trainX #|| output
balanced_count__trainY #|| output

daydiff__trainY #|| output
daydiff__testY #|| output
daydiff__valY #|| output
#%%


