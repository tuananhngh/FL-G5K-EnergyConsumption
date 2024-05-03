"""Analysis of energy consumption for each experiment/host/client/round in FL.

For each experiment in experiment_summary.csv, the energy consumption is computed for each host and the server.
A "energy_hosts_summary.csv" file is created in the experiment folder containing the energy consumption summary for each host.
The energy consumption and parameter summary for each experiment is stored in "perf_summary.csv" in the output directory.
Additional performance results are added to the "perf_summary.csv" file: max_centralized_accuracy, max_distributed_accuracy, round_number.

"""
import pandas as pd 
import os 
from datetime import datetime
import numpy as np

from process_results_for_energy import EnergyResult, match_hosts_estats

def compute_energy_within_range(power_df, start_time, end_time): 
    """
    Computes the energy consumption within a specified time range from a power DataFrame.

    Parameters:
    power_df (DataFrame): The DataFrame containing the power data.
    start_time (str): The start time of the range in the format 'YYYY-MM-DD HH:MM:SS'.
    end_time (str): The end time of the range in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns:
    energy_df (DataFrame): The DataFrame containing the power data within the specified range.
    energy_J (float): The energy consumption in Joules within the specified range.
    energy_kWh (float): The energy consumption in kilowatt-hours within the specified range.
    """
    try:
        energy_df = power_df[(power_df["timestamp"]>=start_time) & (power_df["timestamp"]<=end_time)]
    except TypeError as err:
        print(f"Error: {err}")
        power_df["timestamp"] = pd.to_datetime(power_df["timestamp"], format='mixed')
        energy_df = power_df[(power_df["timestamp"]>=start_time) & (power_df["timestamp"]<=end_time)]
    
    # Compute energy consumption for the round
    intervals = energy_df["timestamp"].diff().apply(lambda x: x.total_seconds())
    energy_J = (energy_df["tot inst power (mW)"] * 1e-3 * intervals).sum()
    energy_kWh = energy_J * 1e-3 / 3600
    return energy_df, energy_J, energy_kWh

def compute_host_energy(power_df, start_datetime, end_datetime):
    """
    Compute the energy consumption for a given host within a specified time range.

    Args:
        energy_df (pandas.DataFrame): DataFrame containing energy data for the host.
        start_datetime (datetime.datetime): Start datetime of the time range.
        end_datetime (datetime.datetime): End datetime of the time range.

    Returns:
        dict: Dictionary containing the computed energy consumption and average resource utilization.

    """
    energy_df, energy_J, energy_kWh = compute_energy_within_range(power_df, start_datetime, end_datetime)
    
    avg_gpu = energy_df["GPU%"].mean()/100
    avg_cpu = energy_df["CPU%"].mean()/100
    avg_ram = energy_df["RAM%"].mean()/100
    
    res = {
        "energy_J": energy_J,
        "energy_kWh": energy_kWh,
        "gpu_perc_avg": avg_gpu,
        "cpu_perc_avg": avg_cpu,
        "ram_perc_avg": avg_ram
    }
    
    return res

def compute_exp_energy_per_host(experiment_summary):
    """
    Compute the energy consumption for each host in an experiment.

    Args:
        summaryfile (pandas.DataFrame): DataFrame containing the experiment summary.
        experiment_summary (dict): Dictionary containing the summary of the experiment.
        experiment_folder (str): Path to the experiment folder.

    Returns:
        pandas.DataFrame: DataFrame containing the energy consumption summary for each host.

    """
    result = EnergyResult(experiment_summary)
    experiment_folder = experiment_summary["result_folder"]
    if not result._folder_still_exist():
        print(f"Folder {experiment_folder} does not exist")
        return None
    server = result._read_server()
    hostname, host_energy = result.client_host_energy()
    
    datetime_format = "%Y-%m-%d_%H-%M-%S_%f"
    start_time = experiment_summary["timestamps.start_experiment"]
    end_time = experiment_summary["timestamps.end_experiment"]
    start_datetime = datetime.strptime(start_time, datetime_format)
    end_datetime = datetime.strptime(end_time, datetime_format)

    host_summary = pd.DataFrame()

    row = compute_host_energy(server.energy, start_datetime, end_datetime)
    row["hostname"] = "server"
    row["role"] = "server"
    row["result_folder"] = experiment_folder
    row["estatsname"] = experiment_summary["server"].split(".")[0].split("-")[1]
    host_summary = pd.concat([host_summary, pd.DataFrame([row])], ignore_index=True)

    for hid, hnrg in enumerate(host_energy):
        row = compute_host_energy(hnrg, start_datetime, end_datetime)
        row["hostname"] = hostname[hid]
        row["result_folder"] = experiment_folder
        row["estatsname"] = match_hosts_estats[hostname[hid]].split("-")[1]
        row["role"] = "client"
        host_summary = pd.concat([host_summary, pd.DataFrame([row])], ignore_index=True)
        
    return host_summary

def compute_exp_energy(config, outputs_path):
    """
    Compute the energy consumption for an experiment.

    Args:
        exp_id (int): Index of the experiment in the summaryfile.
        summaryfile (pandas.DataFrame): DataFrame containing the experiment summaries.
        outputs_path (str): Path to the output directory.

    Returns:
        tuple: A tuple containing the host summary and experiment summary DataFrames.

    """
    experiment_summary = config
    experiment_folder = experiment_summary["result_folder"]
    
    print(f"Processing experiment {experiment_folder}")
    
    host_summary_path = os.path.join(experiment_folder, "energy_hosts_summary.csv")
    # if os.path.exists(host_summary_path):
    #     host_summary = pd.read_csv(host_summary_path)
    # else: 
    host_summary = compute_exp_energy_per_host(experiment_summary)
    if host_summary is None:
        return None, None
    host_summary.to_csv(host_summary_path, index=False)
    
    clients_J, clients_kWh = host_summary[host_summary["role"] == "client"][["energy_J", "energy_kWh"]].sum()
    server_J, server_kWh = host_summary[host_summary["role"] == "server"][["energy_J", "energy_kWh"]].sum()
    
    exp_summary = {
        "result_folder": experiment_folder,
        "clients_J": clients_J,
        "clients_kWh": clients_kWh,
        "server_J": server_J,
        "server_kWh": server_kWh,
        "total_J": clients_J + server_J,
        "total_kWh": clients_kWh + server_kWh
    }
    
    exp_summary = pd.DataFrame(exp_summary, index=[0])
    exp_summary_path = os.path.join(outputs_path, "perf_summary.csv")
    
    exp_summary.to_csv(exp_summary_path, mode='a', index=False, header=not os.path.exists(exp_summary_path))
    
    return host_summary, exp_summary


def add_training_perf(path, summaryfile, perf_path):
    """
    Adds training performance metrics to the summary file.

    Args:
        path (str): The path to the directory containing the summary file.
        summaryfile (pandas.DataFrame): The summary file containing experiment information.
        perf_path (str): The path to save the updated performance summary file.

    Returns:
        pandas.DataFrame: The updated performance summary file.
    """
    if os.path.exists(os.path.join(path,"perf_summary.csv")):
        perf_summary = pd.read_csv(os.path.join(path,"perf_summary.csv"))
    else:
        perf_summary = summaryfile
    # going through each experiment
    for index, row in perf_summary.iterrows():
        # accuracy and nb of rounds
        result = EnergyResult(row)
        server = result._read_server()
        if server.results is None:
            continue
        training_results = {
            "max_centralized_accuracy": server.results["acc_centralized"].max(),
            "max_distributed_accuracy": server.results["acc_distributed"].max(),
            "round_number": server.results["server_round"].max()
        } 
        for key, value in training_results.items():
            perf_summary.loc[index, key] = value
            
        # if correct number host
        host_energy = pd.read_csv(os.path.join(row["result_folder"],"energy_hosts_summary.csv"))
        nb_hosts = len([col for col in summaryfile.columns if "estats" in col])
        perf_summary.loc[index,"nb_hosts"] = nb_hosts
        nb_energy_hosts = len(host_energy) - 1
        perf_summary.loc[index,"nb_energy_hosts"] = nb_energy_hosts
    perf_summary.to_csv(perf_path, index=False)
    return perf_summary
        
def clean(val):
    """
    Cleans the input value by converting it to a float if it is not an instance of float or np.float64.
    To do, we assume that it is a datetime.timedelta object and convert it using the total_seconds() method.
    
    Parameters:
    val (float or np.float64 or datetime.timedelta): The value to be cleaned.

    Returns:
    float: The cleaned value.

    """
    if isinstance(val, float) or isinstance(val, np.float64):
        return val
    else:
        return val.total_seconds()
    
def merge_exp_perf(path):
    """
    Merges the performance summary and experiment summary dataframes based on the 'exp_id' column.
    
    Args:
        path (str): The path to the directory containing the summary files.
        
    Returns:
        pandas.DataFrame: The merged dataframe containing the summary information.
    """
    perf_summary = pd.read_csv(os.path.join(path,"perf_summary.csv"))
    exp_summary = pd.read_csv(
            os.path.join(path,"experiment_summary.csv"),
            parse_dates=[
                "timestamps.end_experiment_after_sleep", 
                "timestamps.end_experiment", 
                "timestamps.start_experiment", 
                "timestamps.start_experiment_before_sleep"
                ],
            date_format='%Y-%m-%d_%H:%M:%S_%f')
    exp_summary["exp_id"] = exp_summary["result_folder"].apply(lambda x: x.split("/")[-1])
    perf_summary["exp_id"] = perf_summary["result_folder"].apply(lambda x: x.split("/")[-1])
    summary = exp_summary.merge(perf_summary, on="exp_id")
    summary["result_folder"] = path + "/" + summary["exp_id"]
    return summary

def merge_client_training_perf(res, hostname):
    """
    Merge the training performance results and evaluation results for a client.

    Args:
        res (object): The object containing the training performance and evaluation results.
        hostname (str): The hostname of the client.

    Returns:
        pandas.DataFrame: The merged DataFrame containing the training performance and evaluation results for the client.
    """
    client_res = res.fittimes.merge(res.fitresults, left_on="Server Round", right_on="server_round")
    drop_col = ["Local Epochs", "Server Round", "LR", "time"]
    client_res = client_res.drop(columns=drop_col)
    client_res["round_role"] = "train"
    cid = client_res["Client ID"].unique()[0]
    res.results.rename(columns={col:"eval_"+col for col in res.results.columns if col!="server_round"}, inplace=True)
    res.results["Client ID"] = cid
    res.results["round_role"] = "eval"
    client_res = pd.concat([client_res, res.results])
    client_res["hostname"] = hostname
    return client_res

def compute_round_stats(training_results):
    """
    Compute statistics for each round of training.

    Args:
        training_results (DataFrame): The training results data.

    Returns:
        DataFrame: The computed statistics for each round of training.
    """
    try:
        fittime_stats = training_results[training_results["round_role"]=="train"].groupby(["hostname", "server_round"])[["fittime"]].agg({"mean", "var", "count"})
    except KeyError as err:
        print("Got error while doing stats in compute_round_stats: "+err)
        print(training_results.columns)
        return None
    fittime_stats.columns = fittime_stats.columns.to_flat_index()
    fittime_stats = fittime_stats.reset_index()
    fittime_stats = training_results[training_results["round_role"]=="train"].merge(fittime_stats, on=["hostname", "server_round"])
    to_merge = fittime_stats.groupby(["server_round"])[[("fittime", "count")]].mean().reset_index().rename(columns={("fittime", "count"):"avg_client_per_host_per_round"})
    fittime_stats = fittime_stats.merge(to_merge, on="server_round")
    fittime_stats["ClientID_host"] = fittime_stats["Client ID"] % 10
    return fittime_stats

def concat_client_training_perf(result, hostmetadata, config):
    """
    Concatenates the training performance data of all clients in the result object.

    Args:
        result (Result): The result object containing the training performance data.
        config (dict): The configuration dictionary containing experiment details.

    Returns:
        DataFrame: The concatenated training performance data with added metadata.

    Raises:
        AttributeError: If the hostname attribute is not found in the hostinfo object.

    """
    training_results = pd.DataFrame()
    for hid in range(len(hostmetadata)):

        hostinfo = result._read_client_host(hid)
        try:
            hostname = hostinfo.hostname
        except AttributeError as err:
            print(f"Error: {err}. Skipping host {hid} and skipping experiment {config['exp_id']}")
            return None
        client_data = hostinfo.clients
        for client in client_data:
            res = client_data[client] 
            client_res = merge_client_training_perf(res, hostname)
            training_results = pd.concat([training_results, client_res])
    
    # compute training round client statistics
    training_results = compute_round_stats(training_results)
    if training_results is None:
        return None
    
    # Add metadata
    training_results["exp_id"] = config["exp_id"]
    training_results = training_results.merge(pd.DataFrame(config).T, on="exp_id")
    return training_results

def aggregate_round_stats(
    logs,
    outputs_paths,
    summary_parquet_file="global_summary.parquet",
    round_parquet_file="round_summary.parquet",
    ):
    """
    Aggregate experiment and round statistics from multiple experiments.

    Args:
        logs (str): Path to the logs directory.
        outputs_paths (list): List of paths to the output directories of each experiment.

    Returns:
        tuple: A tuple containing two DataFrames:
            - results: DataFrame containing the global summary of all experiments.
            - round_results: DataFrame containing the round-wise summary of all experiments.
    """
    results = pd.DataFrame()
    # get experiments metadata
    for path in outputs_paths:
        summary = merge_exp_perf(path)
        results = pd.concat([results, summary], ignore_index=True)
    results["total_kWh"] = results["clients_kWh"] + results["server_kWh"]
    results.drop(columns=["result_folder_y", "result_folder_x"], inplace=True)
    results.to_parquet(os.path.join(logs,summary_parquet_file), index=False)

    round_results = pd.DataFrame()
    for index, config in results.iterrows():
        config_exp_id = config["exp_id"]
        print(f"aggregate_round_stats for {config_exp_id}")
        result = EnergyResult(config)
        hostmetadata = result._get_selectedclient_in_host()
        
        training_results = concat_client_training_perf(result, hostmetadata, config)
        if training_results is None:
            continue
        # Get start and end time of each round
        host_round_time = training_results[["hostname", "server_round", "Client ID", "Start Time", "End Time"]].copy()
        host_round_time = host_round_time.groupby(["hostname", "server_round"]).agg({"Start Time":"min", "End Time":"max"}).reset_index()
        
        # Compute energy consumption for each round
        for hid in range(len(hostmetadata)):
            hostinfo = result._read_client_host(hid)
            try:
                hostname = hostinfo.hostname
            except AttributeError as err:
                print(f"Error: {err}. Skipping host {hid}. But should not reach this line.")
                continue
            
            for round_id in host_round_time[host_round_time["hostname"]==hostname]["server_round"].unique():
                # Get time range for the round and get power timeserie for this time range
                index = host_round_time[(host_round_time["hostname"]==hostname)&(host_round_time["server_round"]==round_id)].index[0]
                start_time, end_time = host_round_time.at[index, "Start Time"], host_round_time.at[index, "End Time"]
                
                _, round_energy_J, round_energy_kWh = compute_energy_within_range(hostinfo.energy, start_time, end_time)
    
                if type(round_energy_kWh) == np.float64:
                    host_round_time.at[index,"round_energy_kWh"] = round_energy_kWh
                    host_round_time.at[index,"round_energy_J"] = round_energy_J
                    
        host_round_time.dropna(inplace=True)
        training_results = training_results.merge(host_round_time.drop(columns=["Start Time", "End Time"]), on=["hostname", "server_round"])

        # clean data
        training_results["round_energy_kWh"] = training_results["round_energy_kWh"].apply(clean)
        training_results["round_energy_J"] = training_results["round_energy_J"].apply(clean)
        
        training_results["round_energy_kWh_per_client"] = training_results["round_energy_kWh"] / training_results["avg_client_per_host_per_round"]
        
        # concat
        round_results = pd.concat([round_results, training_results])
        print("Finished aggregating round stats from experiment "+config_exp_id)
    round_results.to_parquet(os.path.join(logs,round_parquet_file), index=False)
    return results, round_results

def preprocess(paths):
    """
    Preprocesses energy consumption data for each output path.
    Saves resulting dataframes in the output directory.

    Args:
        paths (list): A list of output paths.

    Returns:
        None
    """
    for path in paths:
        summary_path = os.path.join(path,"experiment_summary.csv")
        summaryfile = pd.read_csv(
            summary_path, 
            parse_dates=[
                "timestamps.end_experiment_after_sleep", 
                "timestamps.end_experiment", 
                "timestamps.start_experiment", 
                "timestamps.start_experiment_before_sleep"
                ],
            date_format='%Y-%m-%d_%H:%M:%S_%f'
            )
        summaryfile["exp_id"] = summaryfile["result_folder"].apply(lambda x: x.split("/")[-1])
        summaryfile["result_folder"] = path + "/" + summaryfile["exp_id"]
        for _, config in summaryfile.iterrows():
            compute_exp_energy(config, path)
        # add other performance results
        add_training_perf(path, summaryfile, os.path.join(path,"perf_summary.csv"))
        
if __name__ == "__main__":
    output_paths = [
    # "/home/mjay/energyfl/outputcifar10/fedyogi/labelskew",
    # "/home/mjay/energyfl/outputcifar10/fedavg/labelskew",
    # "/home/mjay/energyfl/outputcifar10/fedadam/labelskew",
    # "/home/mjay/energyfl/outputcifar10/fedadagrad/labelskew",
    "/home/mjay/energyfl/outputcifar10/10clients/fedadagrad/labelskew"
    ]
    preprocess(output_paths)
        