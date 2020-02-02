# Nationwide prediction of type 2 diabetes comorbidities
Source code of analyses conducted for "Nationwide prediction of type 2 diabetes comorbidities" paper published in Nature Scientific Reports.

The source code is kept in the same state as used during submission. Only small changes were made to remove larger sections of unused code.

The analysis was conducted on an airtight server with a greatly limited ability to install software - including python packages. For this reason, it was practical to develop much of the scaffolding code (specifically the pipeline framework) by hand. If time allows, I will publish the pipeline software as a separate package in the future. 

If you have any questions or suggestions, please do not hesitate to reach out.

## Overview of the codebase
The analysis is governed by a pipeline system that reads python source files within `pipeline_tasks` directory extracts appropriately tagged code fragments (tasks) and executes them in sequence. The pipeline keeps track of all tasks inputs and outputs (data objects), which it loads/saves as needed and executes only tasks which outputs are missing. Pipeline parameters as well as task's input data objects are injected into the task namespace as python variables upon tasks start. Additionally, the pipeline logs executed code, standard output as well as a formatted log used to gather facts (`p()` function) throughout the analysis. These facts are later used to populate the publication's numerical values and tables. 
The key design goal of the pipeline system was to enable both an end-to-end execution of the pipeline with varying parameters as well as simple interactive execution of specific tasks (including automatic loading and saving of their results).
The code to large extend uses `#%%` tags/markers, which allow for step by step execution of specific code chunks (jupyter notebook style) in supporting IDEs e.g. Spyder.

## Pipeline
The pipeline code is contained within the file `pipeline_1_1.py` and is very simple. A single pipeline can be executed with different parameters - *pipeline runs*. Each pipeline run has a name, directory where it stores analysis objects specific to that run and an additional directory where it will look for data objects that are shared among multiple runs (e.g. register event dataframes can be generated once and then shared for runs that use the same date of prediction). The file that defines the published analysis is the `T2D__to__OUTCOMES_MIX__v14.py`. Please do not be alarmed by the v14; in the past, I used manual versioning for very minute changes to the pipeline parameters. 
Of note are:
* `execution_plan` - contains a list of tasks to be executed as well as dictionaries which allow mapping tasks inputs to saved objects in case they are different (comes handy if one would like to use multiple patient feature representations but keep using same analysis task for instance). 
* `param_values` - a set of overwritable parameters that defines much of the parameters used throughout the analysis. Specific pipeline runs will overwrite these parameters to specify e.g. outcome ICD code or length of the buffer period.
* What follows is `#%%` separated pipeline run definitions.
* Finally, code at the end of the file makes reports which combine facts gathered through multiple pipeline runs and merge them into a single dictionary that can be saved, compressed and emailed out of the server.

## Tasks
Tasks are extracted automatically on pipeline execution from all python files within the `pipeline_tasks` directory. Each task is composed of:
* header code, shared for all tasks within a given file, composed of all lines of code from the beginning of a given file up to the definition tag of the first task). All variables defined in the header (e.g. `load_input_run_name`) may be overwritten by variables defined in pipeline parameters or input objects.
* task definition code, defining the task (its name and potentially its inputs and outputs), located within two sets of `#||` tags

Additionally, `#|| input` and `#|| output` annotations will mark the first word in that line as an input or output or output object of that task. When a task is executed, lines with `#|| input` annotation will not be executed (as that input object will already have been loaded by the pipeline on tasks start). These annotations are used throughout the code as they enable manual interactive task execution.

# Running the analysis
Forskerservice provides data in SAS format. The `export_all_to_CSV.py` file generated the SAS command to export each table to CSV format. The analysis is run through execution of chunks of `T2D__to__OUTCOMES_MIX__v14.py` file - first the parameter definitions, then the actual pipeline run. The paths within this file will, naturally, need to be adjusted throughout to match the environment.

The analysis, when run on type 2 diabetes population, has significant memory requirement of up to 600GB much of which occurs during feature pivot (transpose events from individual-event-date format to individual-features-counts format) opeartion in *Filter events and individuals* task. This operation's memory footprint could certainly be lowered if the pivot was coded interatively. Additionally, the pivot operation creates a temporary large dataframe index which can cause an integer overflow in pandas for large event frames (10^9++ of observations). For a single pipeline run the intermediate data objects generated will take around 100GB of disk space.

Copying files to shared data storage is not performed automatically. To avoid recalculation of some of the data objects that will not differ between certain pipeline runs (e.g. if the runs differ only in outcome definition data objects generated prior to *Filter events and individuals* task can be re-used) data objects need to be manually moved to appropriate `shared_data_dir_path` directory.
