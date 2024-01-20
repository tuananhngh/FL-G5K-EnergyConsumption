from typing import Dict
import pickle as pkl
import os
import matplotlib.pyplot as plt
from flwr.server import strategy
from torch import mul
from pathlib import Path
# Function read results from path
def read_pkl(path_to_pkl):
    with open(path_to_pkl, "rb") as f:
        result = pkl.load(f)
    return result

def read_result(path_to_last_run, multirun=True):
    ls_results = {}
    if multirun:
        run_dirs = os.listdir(path_to_last_run)
        for rd in run_dirs:
            run_path = os.path.join(path_to_last_run, rd)
            if os.path.isdir(run_path):
                if os.path.exists(os.path.join(run_path, "results.pkl")):
                    pkl_file = os.path.join(run_path, "results.pkl")
                    result = read_pkl(pkl_file)
                    ls_results[rd] = result
    else:
        pkl_file = os.path.join(path_to_last_run, "results.pkl")
        result = read_pkl(pkl_file)
        ls_results["0"] = result
    return ls_results   

def plot_results_multirun(result_file:Dict, metrics)->None:
    #nb_runs = len(result_file)
    #Plot run results on the same figure
    fig,ax = plt.subplots(figsize=(5,5))
    labels = {'0': 'FedOrdered',
              '1': 'FedAvg'}
    for k,v in result_file.items():
        if metrics == "losses_distributed":
            rounds, vals = zip(*v.losses_distributed)
        elif metrics == "losses_centralized":
            rounds, vals = zip(*v.losses_centralized)
        elif metrics == "metrics_centralized":
            rounds, vals = zip(*v.metrics_centralized["accuracy"])
        elif metrics == "metrics_distributed":
            rounds, vals = zip(*v.metrics_distributed["accuracy"])
        ax.plot(rounds,vals, label=labels[k])
    ax.set_xlabel('Communication Round')
    ax.set_ylabel(metrics)
    ax.legend()
    
    

path_to_multirun = "/Users/Slaton/Documents/grenoble-code/fl-flower/jetson-tl/outputs/main_server_0/2024-01-20/21-17-31"
ok = read_result(path_to_multirun, multirun=False)
plot_results_multirun(ok, "losses_centralized")

