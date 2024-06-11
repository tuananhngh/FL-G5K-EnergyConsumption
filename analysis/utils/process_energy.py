import pandas as pd 
import os 
from datetime import datetime
import numpy as np
from box import Box
from .process_experiment import EnergyResults
from .process_data_functions import process_rounds_time_from_log

match_hosts_estats = {
    'host_0': 'estats-11', 
     'host_1': 'estats-12',
     'host_2': 'estats-2',
     'host_3': 'estats-3',
     'host_4': 'estats-4',
     'host_5': 'estats-5',
     'host_6': 'estats-6',
     'host_7': 'estats-7',
     'host_8': 'estats-8',
     'host_9': 'estats-9'
}

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
    energy_J = (energy_df["tot inst power (mw)"] * 1e-3 * intervals).sum()
    energy_kWh = energy_J * 1e-3 / 3600
    gpu_J = (energy_df["gpu inst power (mw)"] * 1e-3 * intervals).sum()
    gpu_kWh = gpu_J * 1e-3 / 3600
    cpu_J = (energy_df["cpu inst power (mw)"] * 1e-3 * intervals).sum()
    cpu_kWh = cpu_J * 1e-3 / 3600
    return energy_df, energy_J, energy_kWh, gpu_J, gpu_kWh, cpu_J, cpu_kWh

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
    energy_df, energy_J, energy_kWh, gpu_J, gpu_kWh, cpu_J, cpu_kWh = compute_energy_within_range(power_df, start_datetime, end_datetime)
    
    avg_gpu = energy_df["gpu%"].mean()/100
    avg_cpu = energy_df["cpu%"].mean()/100
    avg_ram = energy_df["ram%"].mean()/100
    
    res = {
        "energy_J": energy_J,
        "energy_kwh": energy_kWh,
        "gpu_J": gpu_J,
        "gpu_kwh": gpu_kWh,
        "cpu_J": cpu_J,
        "cpu_kwh": cpu_kWh,
        "gpu_perc_avg": avg_gpu,
        "cpu_perc_avg": avg_cpu,
        "ram_perc_avg": avg_ram
    }
    return res

def compute_exp_energy_per_host(results_summary:EnergyResults, strategy:str, epoch:str, exp:int):
    """
    Compute the energy consumption for each host in an experiment.

    Args:
        summaryfile (pandas.DataFrame): DataFrame containing the experiment summary.
        experiment_summary (dict): Dictionary containing the summary of the experiment.
        experiment_folder (str): Path to the experiment folder.

    Returns:
        pandas.DataFrame: DataFrame containing the energy consumption summary for each host.
    """
    experiment_summary = results_summary.get_experiment_summary(strategy, epoch, exp)
    experiment_folder = experiment_summary["result_folder"]
    server = results_summary.get_server_results(strategy, epoch, exp)
    hostname, host_energy = results_summary.get_client_host_energy(strategy, epoch, exp)
    
    start_datetime = experiment_summary["timestamps.start_experiment"]
    #end_datetime = experiment_summary["timestamps.end_experiment"]
    try:
        filtered_info = results_summary.get_perf_filtered_info(strategy, epoch, exp)
        end_datetime = filtered_info["filtered_acc_distributed_time"]
    except KeyError as err:
        print(f"Error: {err}. Get predefined end time.")
        end_datetime = experiment_summary["timestamps.end_experiment"]
    host_summary = pd.DataFrame()
    row = compute_host_energy(server['energy'], start_datetime, end_datetime)
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


def compute_exp_energy_perf(results_summary:EnergyResults, strategy:str, epoch:str, exp:int):
    """
    Compute the energy consumption and performance metrics for an experiment.
    If EnergyResults is filtered, the energy data is also filtered.

    Args:
        exp_id (int): Index of the experiment in the summaryfile.
        summaryfile (pandas.DataFrame): DataFrame containing the experiment summaries.
        outputs_path (str): Path to the output directory.

    Returns:
        tuple: A tuple containing the host summary and experiment summary DataFrames.

    """
    experiment_summary = results_summary.get_experiment_summary(strategy, epoch, exp)
    experiment_folder = experiment_summary["result_folder"]
    #print(exp, experiment_folder)
    print(f"Processing experiment {experiment_folder}")
    
    #host_summary_path = os.path.join(experiment_folder, "energy_hosts_summary.csv")
    # if os.path.exists(host_summary_path):
    #     host_summary = pd.read_csv(host_summary_path)
    # else: 
    host_summary = compute_exp_energy_per_host(results_summary, strategy, epoch, exp)
    if host_summary is None:
        return None, None
    host_summary['strategy'] = [strategy]*len(host_summary)
    host_summary['epoch'] = [epoch]*len(host_summary)
    host_summary['exp_id'] = [exp]*len(host_summary)
    #host_summary.to_csv(f'./files/{strategy}_{epoch}_{exp}_host_summary', index=False)
    
    clients_J, clients_kWh = host_summary[host_summary["role"] == "client"][["energy_J", "energy_kwh"]].sum()
    server_J, server_kWh = host_summary[host_summary["role"] == "server"][["energy_J", "energy_kwh"]].sum()
    # Get server performance
    server = results_summary.get_server_results(strategy, epoch, exp)
    server_results = server['results']
    
    training_results_max = {
        "max_centralized_accuracy": server_results["acc_centralized"].max(),
        "max_centralized_accuracy_round": server_results.loc[server_results['acc_centralized'].idxmax(), 'server_round'],
        "max_distributed_accuracy": server_results["acc_distributed"].max(),
        "max_distributed_accuracy_round": server_results.loc[server_results['acc_distributed'].idxmax(), 'server_round'],
    }
    # Get energy and related metrics
    exp_summary = {
        "result_folder": experiment_folder,
        "clients_J": clients_J,
        "clients_kWh": clients_kWh,
        "server_J": server_J,
        "server_kWh": server_kWh,
        "total_J": clients_J + server_J,
        "total_kWh": clients_kWh + server_kWh,
        "strategy": strategy,
        "epoch": epoch,
        "exp_id": exp,
    }
    
    # Get Filtered Accuracy and Time
    try:
        filtered_infos = results_summary.get_perf_filtered_info(strategy, epoch, exp)
        exp_summary.update(filtered_infos)
    except KeyError as err:
        print(f"Error: {err}. No accuracy filter.")
    exp_summary.update(training_results_max)
    
    #output_path = experiment_folder.split('/')[:-1]
    #output_path = '/'.join(output_path)
    exp_summary = pd.DataFrame(exp_summary, index=[0])
    #exp_summary_path = os.path.join(output_path, "perf_summary.csv")
    #exp_summary.to_csv(exp_summary_path, mode='a', index=False, header=not os.path.exists(exp_summary_path))
    
    return host_summary, exp_summary

def compute_strategy_exp_energy_perf(results_summary:EnergyResults,strategy:str):
    strategies_dict = results_summary.strategies_dict
    perf_strategy_summary = pd.DataFrame()
    host_strategy_summary = pd.DataFrame()
    for epoch in strategies_dict[strategy]['split_epoch']:
        for exp in strategies_dict[strategy]['split_epoch'][epoch]:
            host_summary , exp_perf = compute_exp_energy_perf(results_summary,strategy, epoch, exp)
            perf_strategy_summary = pd.concat([perf_strategy_summary,exp_perf])
            host_strategy_summary = pd.concat([host_strategy_summary,host_summary])
    return host_strategy_summary,perf_strategy_summary

def merge_exp_perf(result_summary:EnergyResults, strategy:str):
    """
    Merges the performance summary and experiment summary dataframes based on the 'exp_id' column.
    Args:
        result_summary (EnergyResults): The object containing the experiment and performance summary data.
        strategy (str): The strategy for which to merge the data.
        
    Returns:
        pandas.DataFrame: The merged dataframe containing the summary information.
    """
    exp_summary = result_summary.summary_file(strategy) #pd DataFrame
    host_strategy_summary, perf_strategy_summary = compute_strategy_exp_energy_perf(result_summary,strategy)
    summary = perf_strategy_summary.merge(exp_summary, on=['result_folder','strategy'])
    return host_strategy_summary, summary

def merge_client_training_perf(res, hostname):
    """
    Merge the training performance results and evaluation results for a client.

    Args:
        res (object): The object containing the training performance and evaluation results.
        hostname (str): The hostname of the client.

    Returns:
        pandas.DataFrame: The merged DataFrame containing the training performance and evaluation results for the client.
    """
    client_res = res.fittimes.merge(res.fitresults, left_on=['server round','lr','local epochs'], right_on=['server_round', 'lr','local_epochs'])
    drop_col = ["local epochs", "server round", 'time']
    client_res = client_res.drop(columns=drop_col)
    client_res["round_role"] = "train"
    cid = client_res["client id"].unique()[0]
    res.results.rename(columns={col:"eval_"+col for col in res.results.columns if col!="server_round"}, inplace=True)
    res.results["client id"] = cid
    res.results["round_role"] = "eval"
    client_res = pd.concat([client_res, res.results])
    #client_res = client_res.merge(res.results, on=["server_round"])
    client_res["hostname"] = hostname
    return client_res


def compute_round_stats(training_results):
    fittime_stats = training_results[training_results["round_role"]=="train"].groupby(["hostname", "server_round"])[["fittime"]].agg({"mean", "var", "count"})
    fittime_stats.columns = fittime_stats.columns.to_flat_index()
    fittime_stats = fittime_stats.reset_index()
    fittime_stats = training_results[training_results["round_role"]=="train"].merge(fittime_stats, on=["hostname", "server_round"])
    to_merge = fittime_stats.groupby(["server_round"])[[("fittime", "count")]].mean().reset_index().rename(columns={("fittime", "count"):"avg_client_per_host_per_round"})
    fittime_stats = fittime_stats.merge(to_merge, on="server_round")
    fittime_stats["ClientID_host"] = fittime_stats["client id"] % 10
    return fittime_stats

def concat_client_training_perf(result_summary:EnergyResults, strategy, epoch, exp)->pd.DataFrame:
    training_results = pd.DataFrame()
    hostmetadata = result_summary.get_clients_in_host(strategy, epoch, exp)
    for hid in range(len(hostmetadata)):
        hostdatafiles = result_summary.get_clients_host_data(strategy, epoch, exp, hid)
        try:
            hostname = hostdatafiles.hostname
        except AttributeError as err:
            print(f"Error: {err}. Skipping host {hid} and skipping experiment {exp}.")
            return None
        client_data = hostdatafiles.clients
        for client in client_data:
            res = client_data[client] 
            client_res = merge_client_training_perf(res, hostname)
            training_results = pd.concat([training_results, client_res])
    # compute training round client statistics
    training_results = compute_round_stats(training_results)
    return training_results

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

def aggregate_round_stats(
    results_summary:EnergyResults,
    summary_parquet_file="global_summary.parquet",
    hosts_summary_parquet_file="hosts_summary.parquet",
    round_parquet_file="round_summary.parquet",
    server_parquet_file="server_round_summary.parquet"
    ):
    """
    This functions should takes all strategies
    Aggregate experiment and round statistics from multiple experiments.

    Args:
        logs (str): Path to the logs directory.
        outputs_paths (list): List of paths to the output directories of each experiment.

    Returns:
        tuple: A tuple containing two DataFrames:
            - results: DataFrame containing the global summary of all experiments.
            - round_results: DataFrame containing the round-wise summary of all experiments.
    """
    # get experiments metadata
    strategies_dict = results_summary.strategies_dict
    strategies = list(strategies_dict.keys())
    results = pd.DataFrame()
    hosts_summary = pd.DataFrame()
    for strategy in strategies:
        print(f"Processing strategy {strategy} in global_summary")
        host_strategy_summary, perf_summary = merge_exp_perf(results_summary, strategy)
        results = pd.concat([results, perf_summary], ignore_index=True)
        hosts_summary = pd.concat([hosts_summary, host_strategy_summary], ignore_index=True)
    results["total_kWh"] = results["clients_kWh"] + results["server_kWh"]
    results.to_parquet(os.path.join('./',summary_parquet_file), index=False)
    hosts_summary.to_parquet(os.path.join('./',hosts_summary_parquet_file), index=False)

    round_results = pd.DataFrame()
    server_round_results = pd.DataFrame()
    for index, config in results.iterrows():
        exp_result_folder = config["result_folder"]
        print(f"aggregate_round_stats for {exp_result_folder}")
        strategy, epoch, exp = config["strategy"], config["epoch"], config["exp_id"]
        hostmetadata = results_summary.get_clients_in_host(strategy, epoch, exp)
        training_results = concat_client_training_perf(results_summary, strategy, epoch, exp) #shape (n_rounds*n_clients, ...)
        #Get server round stats
        server = results_summary.get_server_results(strategy, epoch, exp)
        server_logs = server['logs']
        server_rt = process_rounds_time_from_log(server_logs)
        if training_results is None:
            print(f"Skipping experiment {exp_result_folder}.Training results is None.")
            continue
        # Get start and end time of each round
        host_round_time = training_results[["hostname", "server_round", "client id", "start time", "end time"]].copy()
        host_round_time = host_round_time.groupby(["hostname", "server_round"]).agg({"start time":"min", "end time":"max"}).reset_index() #shape (n_rounds, 3)
        
        # Compute energy consumption for each round
        for hid in range(len(hostmetadata)):
            hostinfo = results_summary.get_clients_host_data(strategy, epoch, exp, hid)
            try:
                hostname = hostinfo.hostname
                #host_machine = match_hosts_estats[hostname]
            except AttributeError as err:
                print(f"Error: {err}. Skipping host {hid}. But should not reach this line.")
                continue
            
            for round_id in host_round_time[host_round_time["hostname"]==hostname]["server_round"].unique():
                # Get time range for the round and get power timeserie for this time range
                index = host_round_time[(host_round_time["hostname"]==hostname)&(host_round_time["server_round"]==round_id)].index[0]
                start_time, end_time = host_round_time.at[index, "start time"], host_round_time.at[index, "end time"]
                try:
                    _, round_energy_J, round_energy_kWh, round_gpu_J, round_gpu_kWh, round_cpu_J, round_cpu_kWh = compute_energy_within_range(hostinfo.energy, start_time, end_time)
                except KeyError as err:
                    print(f"Error: {err}. Skipping host {hid} and round {round_id} of {strategy} {epoch} {exp} {exp_result_folder}.")
    
                if type(round_energy_kWh) == np.float64:
                    host_round_time.at[index,"round_energy_kWh"] = round_energy_kWh
                    host_round_time.at[index,"round_energy_J"] = round_energy_J
                    host_round_time.at[index,"round_gpu_J"] = round_gpu_J
                    host_round_time.at[index,"round_gpu_kWh"] = round_gpu_kWh
                    host_round_time.at[index,"round_cpu_J"] = round_cpu_J
                    host_round_time.at[index,"round_cpu_kWh"] = round_cpu_kWh
                    
                    
        host_round_time.dropna(inplace=True)
        training_results = training_results.merge(host_round_time.drop(columns=["start time", "end time"]), on=["hostname", "server_round"])
        # add strategy, epoch, exp_id information
        for frame in [server_rt, training_results]:
            frame['strategy'] = [strategy]*len(frame)
            frame['epoch'] = [epoch]*len(frame)
            frame['exp_id'] = [exp]*len(frame)
            
        # training_results['strategy'] = [strategy]*len(training_results)
        # training_results['epoch'] = [epoch]*len(training_results)
        # training_results['exp_id'] = [exp]*len(training_results)
        # clean data
        training_results["round_energy_kWh"] = training_results["round_energy_kWh"].apply(clean)
        training_results["round_energy_J"] = training_results["round_energy_J"].apply(clean)
        
        training_results["round_energy_kWh_per_client"] = training_results["round_energy_kWh"] / training_results["avg_client_per_host_per_round"]
        
        # concat
        round_results = pd.concat([round_results, training_results])
        server_round_results = pd.concat([server_round_results, server_rt])
        print("Finished aggregating round stats from experiment "+exp_result_folder)
    round_results.to_parquet(os.path.join('./',round_parquet_file), index=False)
    server_round_results.to_parquet(os.path.join('./',server_parquet_file), index=False)
    return results, hosts_summary, round_results

