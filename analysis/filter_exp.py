import os
import pandas as pd
from box import Box
import json
import re
from typing import Callable, Dict


def read_summaryfile(output_dir:str, usr_homedir:str, condition:Callable = None):
    path_summary = os.path.join(output_dir, "experiment_summary.csv")
    #usr_home = path_summary.replace('/',' ').split()
    #usr_homedir = f"{usr_home[0]}/{usr_home[1]}"
    summary = pd.read_csv(
        path_summary, 
        parse_dates=[
            "timestamps.end_experiment_after_sleep", 
            "timestamps.end_experiment", 
            "timestamps.start_experiment", 
            "timestamps.start_experiment_before_sleep"
            ],
        date_format='%Y-%m-%d_%H-%M-%S_%f')
    summary = match_folder_csv(summary, output_dir)
    # Filter by parameters
    summary["result_folder"] = summary["result_folder"].apply(lambda x: x.replace("/root",usr_homedir))
    if condition is not None: 
        summary = summary.loc[condition(summary)]
        # summary = summary.loc[
        #         ((summary["client.local_epochs"] == 1) & (summary["params.num_rounds"] == 300)) & (summary["client.lr"]==0.0316) |
        #         ((summary["client.local_epochs"].isin([3, 5])) & (summary["params.num_rounds"] == 100)) & (summary["client.lr"]==0.0316)
        #     ]
    # Filter by subfolder
    folder_path = summary["result_folder"].values.tolist()
    for path in folder_path:
        if os.path.isdir(path):
            nb_subfolder = len(os.listdir(path))
            if nb_subfolder == 11:
                continue
            else:
                print(f"Not enough result, remove {path} from summary")
                summary = summary[summary["result_folder"] != path]
    return summary

def match_folder_csv(summaryfile, output_dir):
    correct_file = os.listdir(output_dir)
    summaryfile = summaryfile[summaryfile["result_folder"].apply(lambda x: x.split("/")[-1] in correct_file)]
    return summaryfile

def filter_epochs(summaryfile, epochs_list):
    # Description
    """
    Filter the summary file by epochs and create a dictionary with the path of the experiments
    Args:
    - summaryfile: pd.DataFrame
    - epochs_list: list of int
    Returns:
    - place_holder: Box
    """
    place_holder = Box()
    exp_summary = []
    for e in epochs_list:
        summary_e = summaryfile[summaryfile["client.local_epochs"] == e]
        #print(f"Epoch {e} has {len(summary_e)} experiments")
        if summary_e.shape[0] > 5:
            summary_e = summary_e.iloc[:-1]
        epochs_path = summary_e["result_folder"].values.tolist()
        place_holder[f'epoch_{e}'] = Box(summary=summary_e, path=epochs_path)
        exp_summary.append(summary_e)
    summary_df = pd.concat(exp_summary).reset_index(drop=True)
    return place_holder, summary_df


def create_epochs_dict(by_epochs):
    cols_to_keep_summary = ["result_folder",
                            "server", 
                            "timestamps.start_experiment_before_sleep",
                            "timestamps.start_experiment",
                            "timestamps.end_experiment",
                            "timestamps.end_experiment_after_sleep",
                            "client.local_epochs"]
    epochs_dict = {}
    for epoch in by_epochs.keys():
        byhost = {}
        for i, path in enumerate(by_epochs.__getattr__(epoch).path):
            byhost.setdefault(f'exp_{i}', {})
            params = by_epochs.__getattr__(epoch).summary[by_epochs.__getattr__(epoch).summary["result_folder"] == path]
            params = params[cols_to_keep_summary]
            byhost[f'exp_{i}']['summary'] = params.to_dict(orient='records')[0]        
            subfolder = [(subfold.split('/')[-1], os.path.join(path, f'{subfold}')) for subfold in os.listdir(path)]
            for k in range(len(subfolder)):
                client_name = subfolder[k][0]
                client_name = client_name.replace('client_host','client')
                client_path = subfolder[k][1]
                files = os.listdir(client_path)
                for e,file in enumerate(files):
                    if file == 'client.log' :
                        files[e] = 'client_log'
                    elif file == 'server.log':
                        files[e] = 'server_log'
                    elif file == 'client_pids.csv':
                        files[e] = 'client_pid'
                    else:
                        files[e] = re.split('[._]', file)[0]
                result_files = [(name, os.path.join(client_path,file)) for name,file in zip(files,os.listdir(client_path))]
                for file_name, file_path in result_files:
                    byhost[f'exp_{i}'].setdefault(client_name, {}).setdefault(file_name,file_path)
                #byhost[f'exp_{i}'][client_name] = subfolder[k][1]
        epochs_dict[epoch] = byhost
    return epochs_dict


def create_json_file(strategies, parent_path, usr_homedir:str, condition:Callable, split='labelskew')->Dict[str, Dict[str, Dict[str, Dict]]]:
    strategy_dict = {}
    epoch_list = [1,3,5]
    for strategy in strategies:
        path = os.path.join(parent_path, strategy, split)
        #summary_path = os.path.join(path, "experiment_summary.csv")
        summary = read_summaryfile(path, usr_homedir=usr_homedir, condition=condition)
        by_epochs, summary_epochs = filter_epochs(summary, epoch_list)
        strategy_dict.setdefault(strategy, {}).setdefault('exp_summary', summary_epochs.to_dict(orient='records'))
        strategy_dict[strategy]['split_epoch'] = create_epochs_dict(by_epochs)
        #strategy_dict[strategy] = create_epochs_dict(by_epochs)
    return strategy_dict