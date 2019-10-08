# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:47:30 2018

@author: fskPioDwo
"""
import matplotlib.pyplot as plt

#||
action_name = "Test Task"
action_description = ""
action_input = {"param1", "param2"}
action_output = {"out1"}
#||
#%%
param1 = [5,6,7,8] #|| input
param2 = [-8,7,-6,5] #|| input

out1 = param1 + param2

def plotplot(param1, param2):
    plt.figure()
    plt.plot(param1, param2)
    plt.show() #|| saveplot testplot


    plt.figure()
    plt.plot([10,20,30], [10,0,10])

    plt.figure()
    plt.plot(param1, param1)
    plt.show() #|| saveplot testplot2

print(f"action_instance_name: {action_instance_name} plot_dir_path: {plot_dir_path} tmp_dir_path: {tmp_dir_path} pipeline_action_prefix: {pipeline_action_prefix}")

plotplot(param1, param2)

print(out1)
