import pandas as pd 
import os 
from datetime import datetime

from process_results import EnergyResult, match_hosts_estats, read_summaryfile

def compute_host_energy(energy_df, start_datetime, end_datetime):
    try:
        total_energy_df = energy_df[(energy_df["timestamp"] >= start_datetime) & (energy_df["timestamp"] <= end_datetime)]
    except TypeError as err:
        print(f"Error: {err}")
        energy_df["timestamp"] = pd.to_datetime(energy_df["timestamp"], format='mixed')
        total_energy_df = energy_df[(energy_df["timestamp"] >= start_datetime) & (energy_df["timestamp"] <= end_datetime)]
        
    intervals = total_energy_df["timestamp"].diff().apply(lambda x: x.total_seconds())
    energy_J = (total_energy_df["tot inst power (mW)"] * 1e-3 * intervals).sum()
    energy_kWh = energy_J * 1e-3 / 3600
    avg_gpu = total_energy_df["GPU%"].mean()/100
    avg_cpu = total_energy_df["CPU%"].mean()/100
    avg_ram = total_energy_df["RAM%"].mean()/100
    
    res = {
        "energy_J": energy_J,
        "energy_kWh": energy_kWh,
        "gpu_perc_avg": avg_gpu,
        "cpu_perc_avg": avg_cpu,
        "ram_perc_avg": avg_ram
    }
    
    return res

def compute_exp_energy_per_host(summaryfile, experiment_summary, experiment_folder):
    
    result = EnergyResult(experiment_folder,summaryfile)
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

    row = compute_host_energy(server.__getattribute__("energy"), start_datetime, end_datetime)
    row["hostname"] = "server"
    row["role"] = "server"
    row["result_folder"] = experiment_folder
    row["estatsname"] = experiment_summary["server"].split(".")[0].split("-")[1]
    host_summary = pd.concat([host_summary, pd.DataFrame([row])], ignore_index=True)

    for hid in range(len(host_energy)):
        row = compute_host_energy(host_energy[hid], start_datetime, end_datetime)
        row["hostname"] = hostname[hid]
        row["result_folder"] = experiment_folder
        row["estatsname"] = match_hosts_estats[hostname[hid]].split("-")[1]
        row["role"] = "client"
        host_summary = pd.concat([host_summary, pd.DataFrame([row])], ignore_index=True)
        
    return host_summary

def compute_exp_energy(exp_id, summaryfile, outputs_path):
    
    experiment_summary = summaryfile.to_dict(orient="records")[exp_id]
    experiment_folder = experiment_summary["result_folder"]
    
    print(f"Processing experiment {experiment_folder}")
    
    host_summary_path = os.path.join(experiment_folder, "energy_hosts_summary.csv")
    if os.path.exists(host_summary_path):
        host_summary = pd.read_csv(host_summary_path)
    else: 
        host_summary = compute_exp_energy_per_host(
            summaryfile, experiment_summary, experiment_folder)
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

if __name__ == "__main__":
    output_paths = [
    "/home/mjay/energyfl/outputcifar10/fedyogi/labelskew",
    "/home/mjay/energyfl/outputcifar10/fedavg/labelskew",
    "/home/mjay/energyfl/outputcifar10/fedadam/labelskew",
    "/home/mjay/energyfl/outputcifar10/fedadagrad/labelskew"
    ]
    for path in output_paths:
        summary_path = os.path.join(path,"experiment_summary.csv")
        summaryfile = pd.read_csv(summary_path)
        for exp_id in range(len(summaryfile)):
            compute_exp_energy(exp_id, summaryfile, path)
            
    # add other performance results
    results = pd.DataFrame()
    for path in output_paths:
        summary_path = os.path.join(path,"experiment_summary.csv")
        summaryfile = read_summaryfile(summary_path)
        perf_summary = pd.read_csv(os.path.join(path,"perf_summary.csv"))
        for index, row in perf_summary.iterrows():
            # accuracy and nb of rounds
            result = EnergyResult(row["result_folder"],summaryfile)
            server = result._read_server()
            if server.results is None:
                continue
            training_results = {
                "max_centralized_accuracy": server.results["acc_centralized"].max(),
                "max_distributed_accuracy": server.results["acc_distributed"].max(),
                "round_number": server.results["server_round"].max()
            } 
            for key in training_results.keys():
                perf_summary.loc[index, key] = training_results[key]
            # if correct number host
            host_energy = pd.read_csv(os.path.join(row["result_folder"],"energy_hosts_summary.csv"))
            nb_hosts = len([col for col in summaryfile.columns if "estats" in col])
            perf_summary.loc[index,"nb_hosts"] = nb_hosts
            nb_energy_hosts = len(host_energy) - 1
            perf_summary.loc[index,"nb_energy_hosts"] = nb_energy_hosts
        perf_summary.to_csv(os.path.join(path,"perf_summary.csv"), index=False)