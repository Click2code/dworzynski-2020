import numpy as np
import pickle
from os import path
from time import time
from datetime import datetime, timedelta, date
from pprint import pprint, pformat
from sklearn.base import BaseEstimator
from xgboost.sklearn import XGBClassifier
import json


def load_input(run_name, input_name, shared_dir_name="NOT_EXIST"):
    def check_path(p):
        if path.exists(p + ".pickle"):
            return pickle.load(open(p + ".pickle", "rb"))
        elif path.exists(p+ ".npy"):
            return np.load(p + ".npy")
        else:
            return None

    run_dir_obj = check_path(path.join("V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\", run_name, "data", input_name))
    shared_dir_obj = check_path(path.join("V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_runs\\", shared_dir_name, input_name))
    if (run_dir_obj is None) and (shared_dir_obj is None):
        raise(Exception(f"Couldn't find input file: {input_name}"))
    return run_dir_obj if run_dir_obj is not None else shared_dir_obj



def ready_for_json(d):
    def jsonify(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, dict):
            return ready_for_json(obj)
        if isinstance(obj, (list, tuple)):
            return list([jsonify(list_obj) for list_obj in obj])
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj in (np.NaN, np.NAN, np.NINF, np.nan):
            return "NaN"
        elif isinstance(obj, (datetime, date)):
            return str(obj)
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        elif isinstance(obj, BaseEstimator):
            return obj.__str__
        elif isinstance(obj, XGBClassifier):
            return obj.__str__
        else:
            return obj

    json_ready_dict= {}
    for dkey, dval in d.items():
        #print(dkey)
        json_ready_dict[dkey] = jsonify(dval)

    return json_ready_dict


def ready_for_json_old(d):
    json_ready_dict= {}
    for dkey, dval in d.items():
        if isinstance(dval, dict):
            json_ready_dict[dkey] = ready_for_json(dval)
        elif isinstance(dval, set):
            json_ready_dict[dkey] = list(dval)
        elif isinstance(dval, np.ndarray):
            json_ready_dict[dkey] = dval.tolist()
        elif dval in (np.NaN, np.NAN, np.NINF, np.nan):
            json_ready_dict[dkey] = "NaN"
        elif isinstance(dval, (datetime, date)):
            json_ready_dict[dkey] = str(dval)
        elif isinstance(dval, np.generic):
            json_ready_dict[dkey] = np.asscalar(dval)
        elif isinstance(dval, BaseEstimator):
            json_ready_dict[dkey] = dval.__str__
        elif isinstance(dval, XGBClassifier):
            json_ready_dict[dkey] = dval.__str__
        else:
            json_ready_dict[dkey] = dval
    return json_ready_dict


def dict_to_str(d):
    return json.dumps(ready_for_json(d), separators=(',', ': '), sort_keys=True, indent=4)

stopped_timers = {}
running_timers = {}
def start_stop_timer(timer_name):
    if timer_name in running_timers:
        stopped_timers[timer_name] = time() - running_timers[timer_name]
        print(f"Task \"{timer_name}\" finished in: {stopped_timers[timer_name]} seconds")
    else:
        running_timers[timer_name] = time()
        print(f"\nTask \"{timer_name}\" started at {datetime.now()}")


def my_describe(series, avg_count=10, NA_value=None):
    if NA_value == None:
        filtered_series = series[series.notnull()]
    else:
        filtered_series = series[series != NA_value]
    sorted_filtered_series = filtered_series.sort_values()
    sfseries = sorted_filtered_series

    if avg_count > 0:
        half_avg_count = int(avg_count/2)

        avg_min = np.average(sfseries[:avg_count])
        avg_max = np.average(sfseries[-1 * avg_count:])

        middle_index = int(len(sfseries)/2)
        avg_med = np.average(sfseries[middle_index-half_avg_count:middle_index+half_avg_count])

        first_quarter_index = int(len(sfseries)/4)
        avg_25prcnt = np.average(sfseries[first_quarter_index-half_avg_count:first_quarter_index+half_avg_count])

        last_quarter_index = int(3*len(sfseries)/4)
        avg_75prcnt = np.average(sfseries[last_quarter_index-half_avg_count:last_quarter_index+half_avg_count])
    else:
        avg_min = sfseries[0]
        avg_max = sfseries[-1]

        middle_index = int(len(sfseries)/2)
        avg_med = sfseries[middle_index]

        first_quarter_index = int(len(sfseries)/4)
        avg_25prcnt = sfseries[first_quarter_index]

        last_quarter_index = int(3*len(sfseries)/4)
        avg_75prcnt = sfseries[last_quarter_index]

    return {
        "count_all": int(len(series)),
        "count_not_NA": int(len(sfseries)),
        "mean": float(np.mean(sfseries)),
        "std": float(np.std(sfseries)),
        "avg_min": float(avg_min),
        "avg_25%": float(avg_25prcnt),
        "avg_50%": float(avg_med),
        "avg_75%": float(avg_75prcnt),
        "avg_max": float(avg_max),
    }
