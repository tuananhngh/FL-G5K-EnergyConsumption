import pandas as pd
import numpy as np
import csv
import os
import re
import matplotlib.pyplot as plt

#exp_path = "comm/fedavg/labelskew/2024-04-26_01-39-25"

def load_server_data(exp_path):
    parent_path = os.path.join(exp_path, "server")
    network_log = os.path.join(parent_path, "network.log")
    processes_csv = os.path.join(parent_path, "processes.csv")
    energy_csv = os.path.join(parent_path, "energy.csv")
    server_log = os.path.join(parent_path, "server.log")
    round_time_server = os.path.join(parent_path, "rounds_time.csv")

    # Load data
    processes = pd.read_csv(processes_csv, parse_dates=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
    energy = pd.read_csv(energy_csv, parse_dates=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
    rounds_time = pd.read_csv(round_time_server, parse_dates=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
    network_df = network_log_to_csv(network_log)
    server_log_df = server_log_file_to_csv(server_log)
    return processes, energy, rounds_time, network_df, server_log_df

def load_client_data(exp_path, host_id):
    parent_path = os.path.join(exp_path,f"client_host_{host_id}")
    network_log = os.path.join(parent_path, "network.log")
    client_log = os.path.join(parent_path, "logs.log")
    processes_csv = os.path.join(parent_path, "processes.csv")
    fittimes_csv = os.path.join(parent_path, f"fittimes_client_{host_id}.csv")
    energy_csv = os.path.join(parent_path, f"energy.csv")
    #client_comm = os.path.join(parent_path, f"client_{host_id}_comm.csv")

    # Load data
    processes = pd.read_csv(processes_csv, parse_dates=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
    fittimes = pd.read_csv(fittimes_csv)
    fittimes["Start Time"] = pd.to_datetime(fittimes["Start Time"], format="%Y-%m-%d %H:%M:%S.%f")
    fittimes["End Time"] = pd.to_datetime(fittimes["End Time"], format="%Y-%m-%d %H:%M:%S.%f")
    energy = pd.read_csv(energy_csv, parse_dates=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
    #comm_pd = pd.read_csv(client_comm, parse_dates=["Time"], date_format="%Y-%m-%d %H:%M:%S.%f")
    network_df = network_log_to_csv(network_log)
    client_df = client_log_file_to_pdf(client_log)

    return processes, fittimes, energy, network_df, client_df


def network_log_to_csv(network_log):
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
    data = []
    with open(path_to_log, 'r') as log_file:
        log_lines = log_file.readlines()        
        for line in log_lines:
            match = re.search(r'\[(.*?)\]\[flwr\]\[DEBUG\] - (.*?)_round (\d+)(: (.*))?', line)
            if match:
                timestamp, round_mode, round_number, _, message = match.groups()
                round_number = int(round_number)
                data.append([timestamp, round_mode, round_number, message])
    final_df = pd.DataFrame(data, columns=["timestamp", "round_mode", "round_number", "message"])
    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], format="%Y-%m-%d %H:%M:%S,%f")
    return final_df

def client_log_file_to_pdf(path_to_log):
    data = []
    with open(path_to_log, "r") as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r'\[(.*?)\]\[(.*?)\]\[INFO\] - (CLIENT (\d+) (FIT|END FIT) ROUND (\d+)|Loss: (.*?) \| Accuracy: (.*)|Disconnect and shut down)', line)
            if match:
                timestamp,_,_, client_id, status, round, loss, accuracy = match.groups()
                data.append([timestamp, client_id, status, round, loss, accuracy])
    df = pd.DataFrame(data, columns=['timestamp', 'client_id', 'status', 'round', 'loss', 'accuracy'])
    #df.iloc[-1]["status"] = "Disconnect and shut down"
    df["status"] = df["status"].apply(lambda x: "END EVALUATE" if pd.isnull(x) else x)
    #df["shifted_timestamp"] = df["timestamp"].shift(-1)
    #df["duration"] = (df["shifted_timestamp"] - df["timestamp"]).dt.total_seconds()
    return df


def process_rounds_time(rounds_time, mode='round'):
    """
    Take the rounds_time dataframe and return time taken for each round
    mode : ['round', 'fit', 'eval_dis', 'eval_cen', 'total']
    round : take total duration of each round from fit call to distributed_evaluated_aggregated
    fit : take duration of each round from fit call to fit aggregated
    eval_dis : take duration of each round from distributed evaluate call to distributed evaluated aggregated
    eval_cen : take duration of each round from central evaluate call to central evaluate end
    """
    rounds_time_pivot = rounds_time.pivot(index="round", columns="status", values="timestamp").reset_index()
    rounds_time_pivot.columns = rounds_time_pivot.columns.str.replace(' ','_')
    rounds_time_pivot = rounds_time_pivot.rename(columns={"start_fit_call": "fit_call", 
                                                          "res_fit_aggregated": "fit_agg", 
                                                          "res_fit_received": "fit_received", 
                                                          "central_evaluate_call": "eval_cen_call", 
                                                          "central_evaluated": "eval_cen_received",
                                                          "central_evaluate_end": "eval_cen_agg",
                                                          "distributed_evaluate_call": "eval_dis_call", 
                                                          "distributed_evaluated": "eval_dis_received",
                                                          "distributed_evaluate_end": "eval_dis_agg",})
    status = ["fit","eval_dis","eval_cen"]
    for m in status:
        rounds_time_pivot[f"{m}_duration"] = (rounds_time_pivot[f"{m}_agg"] - rounds_time_pivot[f"{m}_call"]).dt.total_seconds()
        rounds_time_pivot[f"{m}_call_received"] = (rounds_time_pivot[f"{m}_received"] - rounds_time_pivot[f"{m}_call"]).dt.total_seconds()
        rounds_time_pivot[f"{m}_received_agg"] = (rounds_time_pivot[f"{m}_agg"] - rounds_time_pivot[f"{m}_received"]).dt.total_seconds()
    rounds_time_pivot["round_duration"] = (rounds_time_pivot.eval_dis_agg - rounds_time_pivot.fit_call).dt.total_seconds()
    
    if mode == 'round':
        df = rounds_time_pivot[['round', 'fit_call', 'eval_dis_agg','round_duration']]
        df = df.rename(columns={"round":"Server Round","fit_call": "Start Time", "eval_dis_agg": "End Time"})
        return df
    elif mode == 'total':
        df = rounds_time_pivot
        return df
    else:
        df = rounds_time_pivot[['round', f'{mode}_call', f'{mode}_agg',f'{mode}_duration']]
        df = df.rename(columns={"round":"Server Round",f"{mode}_call": "Start Time", f"{mode}_agg": "End Time"})
    return df


def filter_round_time(roundtime_df, tofilter_df):
    """
    Filter the dataframe tofilter_df based on the start and end time in roundtime_df
    """
    tofilter_df["timestamp"] = pd.to_datetime(tofilter_df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
    dfs = []
    for idx, row in roundtime_df.iterrows():
        start_time = row["Start Time"]
        end_time = row["End Time"]
        df = tofilter_df[(tofilter_df["timestamp"] >= start_time) & (tofilter_df["timestamp"] <= end_time)].copy()
        df.loc[:, 'round'] = row["Server Round"]
        df.loc[:, 'round time'] = (end_time - start_time).total_seconds()
        dfs.append(df)
    final_df = pd.concat(dfs)
    return final_df


def plot_send_receive_evolution_round(time_df):
    rounds = time_df["round"].unique()
    sends, receives = [], []
    for round in rounds:
        df = time_df[time_df["round"] == round]
        sends.append(df["send"].sum()/1024)
        receives.append(df["receive"].sum()/1024)
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].plot(rounds, sends, label="Send", marker='^', markevery=1, linewidth=2)
    axs[1].plot(rounds, receives, label="Receive",marker='^', markevery=1, linewidth=2)
    axs[0].set_title("Send Evolution")
    axs[1].set_title("Receive Evolution")
    axs[0].set_ylabel("MB")
    axs[1].set_ylabel("MB")
    plt.show()
    
def plot_send_receive_round(df, round_number=1):
    info_round = df[df["round"] == round_number]
    send = info_round["send"]/1024
    receive = info_round["receive"]/1024
    time = info_round["timestamp"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].bar(time,send, label="Send in KB")
    axs[1].plot(time,receive, label="Receive in KB")
    axs[0].set_title(f"Send in round {round_number}")
    axs[1].set_title(f"Receive in round {round_number}")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    axs[0].set_ylabel("MB/s")
    axs[1].set_ylabel("MB/s")
    plt.show()