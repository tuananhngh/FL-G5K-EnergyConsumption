
# Description: This module contains utility functions to read experiment data from CSV and log files. Many functions requires in argument the path to the experiment directory which is constructed from the experiment dictionnary created by the the class ProcessResults in process_experiment.py.

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from box import Box
import seaborn as sns
import pickle as pkl


def load_server_data(strategies_dict, strategy, epoch, exp, filter=False):
    """
    This function loads server data from various CSV and log files located in the specified experiment path.
    
    Args:
        strategies_dict (dict): A dictionary containing the paths to the server data files.
        strategy (str): The strategy used for the experiment.
        epoch (str): The epoch used for the experiment. e.g. 'epoch_0'.
        exp (str): The experiment number. e.g. 'exp_0'.
        filter (bool): A flag to filter the data from accuracy threshold.
        
    Returns:
        dict: A dictionary containing the loaded server data as pandas DataFrames.
        Dict keys: 'network', 'server_log', 'logs', 'processes', 'energy', 'monitoring' ('round_time' exists exist depend on experiment)
    """
    files = strategies_dict[strategy]['split_epoch'][epoch][exp]['server']
    file_keys = files.keys()
    csv_files = {}
    if filter:
        exp_acc_time = strategies_dict[strategy]['split_epoch'][epoch][exp]['summary']['result_time']['acc_distributed']['time']
    for key in file_keys:
        if key == 'network':
            network_log = files[key]
            file_df = network_log_to_csv(network_log)
        elif key in ['server_log','logs']:
            server_log = files[key]
            file_df = server_log_file_to_csv(server_log)
        else:
            if ('results' not in key) and ('model' not in key):
                file_df = pd.read_csv(files[key])
                file_df.columns = file_df.columns.str.lower()
                for col in file_df.columns:
                    if 'time' in col:
                        try:
                            file_df[col] = pd.to_datetime(file_df[col], format='mixed')
                        except ValueError:
                            print(f"Error : {strategy} {epoch} {exp} {key}, can not converting {col} to datetime ")
        if filter:
            try:
                file_df = file_df[file_df['timestamp'] <= exp_acc_time]
                #print(f"Filtering {key} with timestamp {file_df['timestamp'].iloc[-1]} <= {exp_acc_time}")
            except KeyError as e:
                print(f"KeyError: {strategy} {epoch} {exp} {key} {e}. Skipping Key {key}")
        csv_files[key] = file_df
    return csv_files


def load_client_data(strategies_dict, strategy:str, epoch:str, exp:str, host_id:int, filter=False):
    """
    This function loads client data from various CSV and log files located in the specified experiment path.
    
    Args:
        strategies_dict (dict): A dictionnary object containing the experiment data.
        strategy (str): The strategy used for the experiment.
        epoch (str): The epoch of the experiment. e.g. epoch_1
        exp (str): The experiment number. e.g. exp_0
        host_id (int): The host id of the client. e.g. 0
        filter (bool): A flag to filter the data based on the experiment time. Default is False.
    Returns:
        csv_files: A dictionnary containing pandas DataFrames for processes, fittimes, network, etc. 
    """
    files = strategies_dict[strategy]['split_epoch'][epoch][exp][f"host_{host_id}"]
    file_keys = files.keys()
    csv_files = {}
    if filter:
        exp_acc_time = strategies_dict[strategy]['split_epoch'][epoch][exp]['summary']['result_time']['acc_distributed']['time']
        exp_acc_round = strategies_dict[strategy]['split_epoch'][epoch][exp]['summary']['result_time']['acc_distributed']['round']
        exp_acc = strategies_dict[strategy]['split_epoch'][epoch][exp]['summary']['result_time']['acc_distributed']['acc']
        csv_files['filter_time'] = {'exp_acc_time': exp_acc_time, 'exp_acc_round': exp_acc_round, 'acc': exp_acc}

    for key in file_keys:
        if key == 'network':
            network_log = files[key]
            file_df = network_log_to_csv(network_log)
        elif key in ['client_log','logs']:
            client_logs = files[key]
            file_df = client_log_file_to_pd(client_logs)
        else:
            file_df = pd.read_csv(files[key])
            file_df.columns = file_df.columns.str.lower()
            for col in file_df.columns:
                if 'time' in col:
                    try:
                        file_df[col] = pd.to_datetime(file_df[col], format="mixed")
                    except ValueError:
                        print(f"Error : {strategy} {epoch} {exp} host_{host_id} {key}, can not converting {col} to datetime: {file_df[col].iloc[0]}")
        if filter:
            try:
                if 'server round' in file_df.columns:
                    file_df = file_df[file_df['server round'] <= exp_acc_round]
                    #print(f"Filtering {key} with server round")
                elif 'timestamp' in file_df.columns:
                    file_df = file_df[file_df['timestamp'] <= exp_acc_time]
                    #print(f"Filtering {key} with timestamp {file_df['timestamp'].iloc[-1]} <= {exp_acc_time}")
                elif 'time' in file_df.columns:
                    file_df = file_df[file_df['time'] <= exp_acc_time]
                    #print(f"Filtering {key} with time {file_df['time'].iloc[-1]} <= {exp_acc_time}")
            except KeyError as e:
                print(f"KeyError: {strategy} {epoch} {exp} {host_id} {key} {e}")    
        csv_files[key] = file_df
    return csv_files



def network_log_to_csv(network_log):
    """
    This function reads a network log file and converts it into a pandas DataFrame.
    
    Args:
        network_log (str): The path to the network log file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the timestamp, process name, send and receive data from the network log.
    """
    data = []
    with open(network_log, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("2024"):
                try:
                    date, time, process, send, receive = line.split(maxsplit=4)
                    if process.startswith("python3"):
                        timestamp = f"{date} {time}"
                        send = float(send)
                        receive = float(receive)
                        timestamp = pd.to_datetime(timestamp, format="%Y-%m-%d %H:%M:%S.%f")
                        data.append([timestamp, process, send, receive])
                except ValueError:
                    continue
    return pd.DataFrame(data, columns=["timestamp", "process", "send", "receive"])


def server_log_file_to_csv(path_to_log):
    """
    This function reads a server log file and converts it into a pandas DataFrame.
    """
    data = []
    with open(path_to_log, 'r') as log_file:
        log_lines = log_file.readlines()        
        for line in log_lines:
            match = re.search(r'\[(.*?)\]\[flwr\]\[DEBUG\] - (.*?)_round (\d+)((.*))?', line)
            if match:
                timestamp, round_mode, round_number, message,_ = match.groups()
                round_number = int(round_number)
                message = message.replace(':',' ') if ':' in message else message
                data.append([timestamp, round_mode, round_number, message])
    final_df = pd.DataFrame(data, columns=["timestamp", "round_mode", "round_number", "message"])
    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], format="%Y-%m-%d %H:%M:%S,%f")
    return final_df

def client_log_file_to_pd(path_to_log):
    """
    This function reads a client log file and converts it into a pandas DataFrame.
    """
    data = []
    with open(path_to_log, "r") as f:
        lines = f.readlines()
        for line in lines:
            if 's/it]' in line:
                # Split the line at the first occurrence of '][' and keep the second part
                line = line.split('][', 1)[-1]
            match = re.search(r'\[(.*?)\]\[(.*?)\]\[INFO\] - (CLIENT (\d+) (FIT|END FIT) ROUND (\d+)|Loss: (.*?) \| Accuracy: (.*)|Disconnect and shut down)', line)
            if match:
                timestamp,_,_, client_id, status, round, loss, accuracy = match.groups()
                try:
                    timestamp = pd.to_datetime(timestamp, format="%Y-%m-%d %H:%M:%S,%f")
                except:
                    print(f"Error: {timestamp} Line: {line}")
                data.append([timestamp, client_id, status, round, loss, accuracy])
    df = pd.DataFrame(data, columns=['timestamp', 'client_id', 'status', 'round', 'loss', 'accuracy'])
    #df.iloc[-1]["status"] = "Disconnect and shut down"
    df["status"] = df["status"].apply(lambda x: "END EVALUATE" if pd.isnull(x) else x)
    #df["shifted_timestamp"] = df["timestamp"].shift(-1)
    #df["duration"] = (df["shifted_timestamp"] - df["timestamp"]).dt.total_seconds()
    return df


def read_server_clients_data(exp_path):
    """
    This function reads the server and clients data from the specified experiment path.
    Args:
        exp_path (str): The path to the experiment directory.
    Returns:
        Box: A Box object containing the server and clients data as pandas DataFrames.
        process: The processes data.
        energy: The energy data.
        time: The time data.
        network: The network data.
        log: The log data.
    """
    host_ids = [int(name.split('_')[-1]) for name in os.listdir(exp_path) if 'client' in name]
    outputs = Box()
    for i in host_ids:
        client_processes, client_fittimes, client_energy, client_network, client_df = load_client_data(exp_path, i)
        outputs[f'client_{i}'] = Box(processes=client_processes, energy=client_energy, time=client_fittimes, network=client_network, log=client_df)
    server_path = os.path.join(exp_path, 'server')
    server_processes, server_energy, server_time, server_network, server_df = load_server_data(exp_path)
    outputs['server'] = Box(processes=server_processes, energy=server_energy, time=server_time, network=server_network, df=server_df)
    return outputs


def flwr_pkl(path_to_pkl):
    """
    Load and return the contents of a pickle file.

    Parameters:
    path_to_pkl (str): The path to the pickle file.

    Returns:
    object: The deserialized object from the pickle file.
    """
    with open(path_to_pkl, "rb") as f:
        result = pkl.load(f)
    return result
    

