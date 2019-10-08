# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:18:34 2017

@author: fskPioDwo
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 20:35:18 2017

@author: fskPioDwo
"""
import logging
from pprint import pformat
from os import path, W_OK, listdir
import os
import time
import sys
from glob import glob
import re
from zipfile import ZipFile
#import cPickle as pickle
import pickle
import networkx as nx
import pandas as pd
import io
import contextlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json

logging.basicConfig(level=logging.INFO)
debug = logging.debug
info = logging.info
warn = logging.warn
err = logging.error

class InputOutputObject:

    name = None
    value = None
    f_path = None


    def __init__(self, name, value__or__f_path):
        self.name = name

        def looks_like_a_path(s):
            #In order to be a path you need to be a string, have at lest one \ before file name and have an extension
            return type(s) == str and re.match("^([A-Z]:)?\.?(\\\\[A-Za-z0-9 _\-\.]+)*\\\\[A-Za-z_\-0-9 ]+(\.[a-zA-Z0-9]+)(\.zip)?$", s) != None

        if looks_like_a_path(value__or__f_path):
            self.f_path = value__or__f_path
        else:
            self.value = value__or__f_path

    def read_pickle(self, f):
        return pickle.load(f)

    def read_csv(self, f):
        return pd.read_csv(f)

    def read_png(self, f):
        return mpimg.imread(f)

    def read_npy(self, f):
        return np.load(f)

    def save(self, data_dir_path, suffix=""):
        assert(self.value is not None)

        def get_fp(data_dir_path, extension):
            return os.path.join(data_dir_path, self.name + "__" + suffix +"." + extension if suffix != "" else self.name+"."+extension)

        assert(self.f_path == None)

        if hasattr(self.value, "to_pickle"):
            self.f_path = get_fp(data_dir_path, "pickle")
            self.value.to_pickle(self.f_path)
        elif type(self.value) == np.ndarray:
            self.f_path = get_fp(data_dir_path, "npy")
            np.save(self.f_path, self.value)
        else:
            self.f_path = get_fp(data_dir_path, "pickle")
            with open(self.f_path, "wb") as f:
                pickle.dump(self.value, f, protocol=-1)

    def read_file(self):
        file_associations = {
            "csv": self.read_csv,
            "pickle": self.read_pickle,
            "png": self.read_png,
            "npy": self.read_npy
            #"zip": self.read_zip
        }

        f_extension = self.f_path.split(".")[-1]
        if f_extension not in file_associations:
            raise Exception("Unknown file extension in file: {}, parsed extension: {}".format(self.f_path, f_extension))

        with open(self.f_path, "rb") as f:
            return file_associations[f_extension](f)

    def get_value(self):
        if self.value is None:
            self.value = self.read_file()
        return self.value


class Action:
    """
    input = {
        "name": InputOutputObject
    }
    """
    name = ""
    description = ""

    input_var_names = []
    output_var_names = []

    setup_code = ""
    code = None

    __locals = {}
    __globals = {}

    def __init__(self, name, description, input_var_names, output_var_names, setup_code, code):
        self.name = name
        self.description = description
        self.input_var_names = input_var_names
        self.output_var_names = output_var_names
        self.setup_code = setup_code
        self.code = code

    @staticmethod
    def parse_script(file_path):
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()

        #Regex definitions
        lone_marker = re.compile("^#\|\|\s*$") #re.match("^#\|\|$", "#||\n")
        io_marker = re.compile("^([a-zA-Z0-9_]*)[\s=\(].*#\|\|\s*(input|output)(\s+([a-zA-Z0-9_]+))?") #re.match("^([a-zA-Z0-9_]*)[\s=\(\[].*#\|\|\s?(input|output)", "var1 = 13 #|| input")

        #
        #Divide file into sections
        #
        sections = {
            "header": (None, None),
            "actions": []
            #step would look like:
            #definition: (start_line_i, stop_line_i)
            #code: (start_line_i, stop_line_i)
            #input: [line_i, ...]
            #output: [line_i, ...]
        }

        #Find header (headre is the part where most includes happen - shared across actions)
        #Find action definitions
        current_section = "header" # header | definition | code
        header_start_line_i = 0
        header_end_line_i = None
        action_definition_start_line_i = None
        action_definition_end_line_i = None
        action_code_start_line_i = None
        action_code_end_line_i = None
        for line_i, line in enumerate(lines):
            lone_marker_match = lone_marker.match(line)
            if lone_marker_match and current_section == "header":
                current_section = "definition"
                action_definition_start_line_i = line_i+1
                header_end_line_i = line_i-1
                sections["header"] = (header_start_line_i, header_end_line_i)
            elif lone_marker_match and current_section == "definition":
                current_section = "code"
                action_definition_end_line_i = line_i-1
                action_code_start_line_i = line_i+1
            elif current_section == "code" and (lone_marker_match or line_i == len(lines) - 1):
                current_section = "definition"
                action_code_end_line_i = line_i-1
                sections["actions"].append(
                    {"definition": (action_definition_start_line_i, action_definition_end_line_i),
                     "code": (action_code_start_line_i, action_code_end_line_i),
                     "input": [], "output": []}
                )
                action_definition_start_line_i = line_i+1 #this is for the next action
        debug(pformat(sections))

        if len(sections["actions"]) == 0:
            return {}

        #Find input and output variables
        for action in sections["actions"]:
            action["special_line_numbers"] = {"input": [], "output": []}

            for line_i in range(action["code"][0], action["code"][1]+1):
                input_marker_match = io_marker.match(lines[line_i])
                if input_marker_match:
                    var_name, iotype, ignore, suffix = input_marker_match.groups()
                    action[iotype].append(var_name)
                    action["special_line_numbers"][iotype].append(line_i)

        header_code = "".join(lines[sections["header"][0]: sections["header"][1]+1])

        parsed_actions = {}
        #Parse description
        for action in sections["actions"]:
            extracted_vars = {}
            exec("".join(lines[action["definition"][0]: action["definition"][1]+1]), {}, extracted_vars)
            action_name = extracted_vars["action_name"]
            action_description = extracted_vars["action_description"]
            if "action_input" in extracted_vars:
                action["input"].extend(extracted_vars["action_input"])
            if "action_output" in extracted_vars:
                action["output"].extend(extracted_vars["action_output"])

#            action_code_lines = lines[action["code"][0]: action["code"][1]+1]
#            for line_i_to_mask in action["special_line_numbers"]["input"]:
#                relative_line_i = line_i_to_mask - action["code"][0]
#                action_code_lines[relative_line_i] = "#PIPELINE MASKED: " + action_code_lines[relative_line_i]
            for line_i in range(action["code"][0], action["code"][1]+1):
                if line_i in action["special_line_numbers"]["input"]:
#                    relative_line_i = line_i - action["code"][0]
                    lines[line_i] = "#PIPELINE MASKED: " + lines[line_i]

            for line_i in range(action["code"][0], action["code"][1]+1):
                line = lines[line_i]
                if line.strip() == "plt.show()":
                    lines[line_i] = "#PIPELINE MASKED: " + line
                if "#|| saveplot" in line:
                    #find first non-white char in the line
                    line_whitespace_prefix = ""
                    for c in line:
                        if c not in " \t":
                            break
                        line_whitespace_prefix += c
                    #Read saveplot arguments
                    saveplot_argument_string = line[line.find("#|| saveplot") + len("#|| saveplot"):]
                    saveplot_arg_name = saveplot_argument_string.strip()
                    #saveplot_arguments = saveplot_argument_string.strip().split()
                    #assert(len(saveplot_arguments))
                    #saveplot_arg_name = saveplot_arguments[0]
                    #saveplot_arg_vertprop = saveplot_arguments[1] if len(saveplot_arguments) > 1 else None
                    savefig_command = f"plt.savefig(plot_dir_path + '\' + action_instance_name + '__{saveplot_arg_name}.png', dpi=96, bbox_inches='tight') #PIPELINE INSERTED"
                    #replace line with the savefig command
                    lines[line_i] = line_whitespace_prefix + savefig_command

            action_code = "".join(lines[action["code"][0]: action["code"][1]+1])

            action = Action(action_name, action_description, action["input"], action["output"], setup_code=header_code, code=action_code)
            parsed_actions[action_name] = action

            info(f'{action_name}:\n\tdescription: {action.description}\n\tinputs: {",".join(action.input_var_names)}\n\toutputs: {",".join(action.output_var_names)}')

        return parsed_actions

def graph_name_to_var_name(gn):
    return gn.split(".")[-1]

def var_name_to_graph_name(vn, action_name):
    return action_name + ".out." + vn

from itertools import chain
class PipelineRun:

    requirements = {}
    execution_plan = None

    def __init__(self, name, pipeline, target, param_values, pipeline_dir_path, shared_data_dir_path=None):
        self.name = name
        self.pipeline = pipeline
        self.target = target
        self.param_values = param_values
        self.shared_data_dir_path = shared_data_dir_path

        self.run_dir_path = path.join(pipeline_dir_path, self.name)
        self.data_dir_path = os.path.join(self.run_dir_path, "data")
        self.plot_dir_path = os.path.join(self.run_dir_path, "plots")
        self.tmp_dir_path = os.path.join(self.run_dir_path, "tmp")
        self.exported_values_dir_path = os.path.join(self.run_dir_path, "values")

    def get_file_name(self, requirement_name):
        return requirement_name + ".ppickle"

    def get_requirement_name_from_file_name(self, file_name):
        m = re.match("(.*\\\\)?([a-zA-Z0-9 _\-\.]+)\.(ppickle|pickle|npy)", file_name)
        return m.groups()[-2]

    def gather_available_saved_data(self):
        files_in_data_dir = [path.join(self.data_dir_path, fp) for fp in os.listdir(self.data_dir_path) if path.isfile(path.join(self.data_dir_path, fp))]
        dirs_in_data_dir = [path.join(self.data_dir_path, fp) for fp in os.listdir(self.data_dir_path) if path.isdir(path.join(self.data_dir_path, fp))]
        for dir_path in dirs_in_data_dir:
            files_in_data_dir.extend([path.join(dir_path, fp) for fp in os.listdir(dir_path) if path.isfile(path.join(dir_path, fp))])
        if self.shared_data_dir_path != None:
            files_in_data_dir.extend([path.join(self.shared_data_dir_path, fp) for fp in os.listdir(self.shared_data_dir_path) if path.isfile(path.join(self.shared_data_dir_path, fp))])
        #saved_data_objects = [InputOutputObject(name=self.get_requirement_name_from_file_name(fp), value__or__f_path=fp) for fp in files_in_data_dir if "summary"not in fp] #To allow for rerunning of describe actions
        saved_data_objects = [InputOutputObject(name=self.get_requirement_name_from_file_name(fp), value__or__f_path=fp) for fp in files_in_data_dir if ("calibration" not in fp) and ("summary" not in fp)]
        return saved_data_objects

    def execute(self):
        #Create run dir if doesn't exist
        for dir_path in [self.run_dir_path, self.data_dir_path, self.plot_dir_path, self.tmp_dir_path, self.exported_values_dir_path]:
            if not path.exists(dir_path):
                os.mkdir(dir_path)

        #Look for available data
        available_saved_data = self.gather_available_saved_data()
        provided_data = [InputOutputObject(name=p_name, value__or__f_path=p) for p_name, p in list(self.param_values.items())]
        satisfied_requirements = {io_obj.name: io_obj for io_obj in chain(provided_data, available_saved_data)} #Note that data actually saved will override provided data
        info('Available provided data:\n{}'.format("\n\t".join(sorted([io_obj.name for io_obj in provided_data]))))
        info('Available saved data:\n{}'.format("\n\t".join(sorted([io_obj.name for io_obj in available_saved_data]))))
        self.pipeline.execute_plan(self.name, satisfied_requirements, self.run_dir_path, self.data_dir_path, self.plot_dir_path, self.tmp_dir_path, self.exported_values_dir_path)

import traceback

@contextlib.contextmanager
def stdoutRedirectIO(stdout):
    stdout_orig = sys.stdout
    sys.stdout = stdout
    yield stdout
    sys.stdout = stdout_orig


def pipeline_exec(code, global_env, local_env, logger):
    exc_type, exc_obj, tb = None, None, None
    with stdoutRedirectIO(logger) as l:
        try:
            exec(code, global_env, local_env)
        except Exception:
            #traceback.print_exc()
            exc_type, exc_obj, tb = sys.exc_info()
    if tb != None:
        print("Something went wrong in the code")
        print(len(code.split("\n")))
        #print(tb.tb_lineno) #This is not thre right line. It will always point to code exec
        #print(code.split("\n")[tb.tb_lineno])
        for line in traceback.format_exception(exc_type, exc_obj, tb):
          sys.stderr.write(line + "\n")
          m = re.match("File \"<string>\", line ([0-9]*),.*", line)
          if m:
              line_number_in_action = int(m.groups()[0])
              sys.stderr.write("Line in question: {}".format(code.split("\n")[line_number_in_action]))
#        sys.stderr.write("\n".join(traceback.format_exception(exc_type, exc_obj, tb)))
        return False
    return True



class PipelineLogger():
    def __init__(self, action_instance_name, run_dir_path):
        self.terminal = sys.stdout
        self.logfile = open(path.join(run_dir_path, "{}__stdout.txt".format(action_instance_name)), "w")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.flush()

    def flush(self):
        self.logfile.flush()
        self.terminal.flush()

    def close(self):
        self.flush()
        self.logfile.close()


class Pipeline:

    action_templates = {}
    graph = None
    action_graph = None
    output_data_objs = {}

    def __init__(self, script_re, execution_plan):
        scripts = glob("./{}".format(script_re))
        for script_fp in scripts:
            extracted_actions = Action.parse_script(script_fp)
            self.action_templates.update(extracted_actions)
            info("Parsing: {} extracted actions: {}".format(script_fp, len(extracted_actions)))

        self.execution_plan = execution_plan
        self.actions_template_map = {action_instance_name: self.action_templates[action_t_name] for action_t_name, action_instance_name, x, y in execution_plan}
        #self.__assemble_execution_graph(execution_plan)

    def __assemble_execution_graph(self, execution_plan):
        self.graph = nx.DiGraph()
        self.action_graph = nx.DiGraph() #Directed graph

        for action_name, action_instance_name, action_instance_prefix, action_bindings in execution_plan:
            action = self.action_templates[action_name]

            self.graph.add_node(action_instance_name, type="action", action=action)
            self.action_graph.add_node(action_instance_name, type="action", action=action)

            for input_name in action.input_var_names:
                node_name = action_instance_name + ".in." + input_name
                self.graph.add_node(node_name, type="obj", obj_name=input_name)
                self.graph.add_edge(node_name, action_instance_name)
                if input_name in action_bindings:
                    #print(input_name)
                    #print(self.output_data_objs)
                    assert(action_bindings[input_name] in self.output_data_objs)
                    self.graph.add_edge(self.output_data_objs[action_bindings[input_name]], node_name)
                    self.action_graph.add_edge(self.output_data_objs[action_bindings[input_name]].split(".")[0], action_instance_name)
                else:
                    if input_name in self.output_data_objs:
                        self.graph.add_edge(self.output_data_objs[input_name], node_name)
                        self.action_graph.add_edge(self.output_data_objs[input_name].split(".")[0], action_instance_name)

            for output_name in action.output_var_names:
                instance_output_name = output_name if action_instance_prefix == "" else action_instance_prefix + "__" + output_name
                node_name = action_instance_name + ".out." + instance_output_name
                self.graph.add_node(node_name, type="obj", obj_name=output_name)
                self.graph.add_edge(action_instance_name, node_name)
                #print(output_name)
                #print(self.output_data_objs)
                assert(instance_output_name not in self.output_data_objs)
                #self.output_data_objs[output_name] = instance_output_name
                self.output_data_objs[instance_output_name] = node_name


    def execute_plan(self, run_name, available_data, run_dir_path, data_dir_path, plot_dir_path, tmp_dir_path, exported_values_dir_path):

        def prefix_name_if_needed(var_name, action_prefix):
            return var_name if action_prefix == "" else action_prefix + "__" + var_name

        def pipeline_force_save_obj_factory(data_dir_path):
            pipeline_force_save_obj_func = lambda obj, obj_name: InputOutputObject(prefix_name_if_needed(obj_name), obj).save(data_dir_path)
            return pipeline_force_save_obj_func

        for action_template_name, action_instance_name, action_prefix, mapped_variables in self.execution_plan:
            self.temp = []
            print("\n\n########################################")
            print(f"\nFor run: {run_name}")
            print(("\nPreparing to execute: {}".format(action_instance_name)))
            action = self.action_templates[action_template_name]



            print("\nEvaluating pre existing output")
            missing_output_vars = list([var_name for var_name in action.output_var_names if prefix_name_if_needed(var_name, action_prefix) not in available_data])
            if len(missing_output_vars) == 0 and len(action.output_var_names) > 0:
                print("All variables present, no need to run")
                continue
            else:
                print(("Missing: {}".format(missing_output_vars)))

            logger = PipelineLogger(action_instance_name, run_dir_path)

            start = time.time()
            print("\nLoading input")

            action_value_store = {"inputs": {}}

            action_global = {"plt": plt, "action_instance_name": action_instance_name, "plot_dir_path": plot_dir_path, "value_store": action_value_store, "tmp_dir_path": tmp_dir_path, "pipeline_action_prefix": action_prefix}
            action_local = {}
            for input_var_name in action.input_var_names:

                #If the variable is mapped from one name to another
                if input_var_name in mapped_variables:

                    mapped_var_name = mapped_variables[input_var_name]

                    if mapped_var_name not in available_data:
                        raise Exception(f"Missing mapped variable: {mapped_var_name} from available data")

                    mapped_variable = available_data[mapped_var_name]

                    #Keep it for future reference
                    action_value_store["inputs"][input_var_name] = \
                        mapped_variable.f_path if mapped_variable.f_path != None else mapped_variable.value

                    action_global[input_var_name] = mapped_variable.get_value()



                #In all other cases (default)
                else:
                    if input_var_name not in available_data:
                        raise Exception("Couldn't find input var: {}".format(input_var_name))

                    variable = available_data[input_var_name]

                    #Keep it for future reference
                    action_value_store["inputs"][input_var_name] = variable.f_path if variable.f_path != None else variable.value

                    action_global[input_var_name] = variable.get_value()
                #print("injecting: {}".format(input_var_name))
            print(("Loaded Global environment: {}".format(list(action_global.keys()))))


            pipeline_exec(action.setup_code, action_global, action_global, logger) #global is here twice as during exec python apparently only assigns to local and things like libs should be global

            with open(run_dir_path + "\\" + action_instance_name + "__code.txt", "w") as f:
                f.write(action.code)

            start = time.time()
            #print(action.code)
            print(("\nRunning action: {}".format(action_instance_name)))
            went_ok = pipeline_exec(action.code, action_global, action_global, logger)
            print(("Action \"{}\" completed in {}".format(action_instance_name, time.time() - start)))
            if not went_ok:
                print("There was an error during execution. Aborting.")
                return None

            start = time.time()
            print("\nSaving output")
            for output_var_name in action.output_var_names:
                io_obj = InputOutputObject(name=prefix_name_if_needed(output_var_name, action_prefix),
                                           value__or__f_path=action_global[output_var_name])
                self.temp.append(io_obj)

            for io_obj in self.temp:
                print(("Saving: {}".format(io_obj.name)))
                io_obj.save(data_dir_path=data_dir_path)
                available_data[io_obj.name] = io_obj
            print(("Output saved in {}".format(time.time() - start)))

            start = time.time()
            print("\nSaving exported values")
            pickle.dump(action_value_store, open(path.join(exported_values_dir_path, action_instance_name + ".pickle"), "wb"))
            print(("Exported values (len: {}) saved in {}".format(len(action_value_store), time.time() - start)))

            print("\nSaving figures")
            print(f"Number of figures left: {plt.get_fignums()}")
            for fig_index in plt.get_fignums():
                fig = plt.figure(fig_index)
                fig.savefig(path.join(plot_dir_path, "{}__{}.png".format(action_instance_name, fig_index)), transparent=True, dpi=96)
                plt.close(fig_index)

            logger.close()
#%%
execution_plan = [
        ("Test Task", "Test Task", "", {}),
]


param_values = {
        "param1": [1,2,3,4],
        "param2": [-1,-2,-3,-4]
}


p = Pipeline("pipeline_tasks\*.py", execution_plan)

#%%

r = PipelineRun("test_pipeline", p, target="trainY", param_values=param_values, pipeline_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_test_runs\\",
                shared_data_dir_path="V:\\Projekter\\FSEID00001620\\Piotr\\pipeline_test_runs\\shared_folder")
r.execute()

