
from typing import Dict, List, Tuple, Callable, Optional
from types import SimpleNamespace
import pickle as pkl
import re
import os
import yaml # pip install PyYAML
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from box import Box

#Create a class grouping all the functions that read csv and log files. 

energy_cols = ['timestamp',
 'RAM%',
 'GPU%',
 'GPU inst power (mW)',
 'GPU avg power (mW)',
 'CPU%',
 'CPU inst power (mW)',
 'CPU avg power (mW)',
 'tot inst power (mW)',
 'tot avg power (mW)']

evalresult_cols = ['time', 'server_round', 'loss', 'accuracy']
fitresult_cols = ['time', 'server_round', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
fittimes_cols =['Client ID', 'Server Round', 'Start Time', 'End Time', 'fittime']

match_hosts_estats = {
    'client_host_0': 'estats-11', 
     'client_host_1': 'estats-12',
     'client_host_2': 'estats-2',
     'client_host_3': 'estats-3',
     'client_host_4': 'estats-4',
     'client_host_5': 'estats-5',
     'client_host_6': 'estats-6',
     'client_host_7': 'estats-7',
     'client_host_8': 'estats-8',
     'client_host_9': 'estats-9'
}

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

class ProcessResults:
    def __init__(self, 
                 output_dir:str, 
                 usr_homedir:str, 
                 strategies:List[str],
                 condition:Callable=None, 
                 threshold:Optional[float]=0.75,
                 epoch_list:List[int] = [1,3,5],
                 split='labelskew'):
        self.output_dir = output_dir
        self.usr_homedir = usr_homedir
        self.strategies = strategies
        self.condition = condition
        self.threshold = threshold
        self.epoch_list = epoch_list
        self.split = split
        self.strategies_dict = self._create_json_file()
        
    def read_summaryfile(self, strategy_path:str):
        path_summary = os.path.join(strategy_path, "experiment_summary.csv")
        summary = pd.read_csv(
            path_summary, 
            parse_dates=[
                "timestamps.end_experiment_after_sleep", 
                "timestamps.end_experiment", 
                "timestamps.start_experiment", 
                "timestamps.start_experiment_before_sleep"
                ],
            date_format='%Y-%m-%d_%H-%M-%S_%f')
        summary = self._match_folder_csv(summary, strategy_path)
        print(f"Summary shape before matching condition : {summary.shape}")
        # Filter by parameters
        summary["result_folder"] = summary["result_folder"].apply(lambda x: x.replace("/root",  self.usr_homedir))
        if self.condition is not None: 
            summary = summary.loc[self.condition(summary)]
        # Filter by subfolder
        #folder_path = summary["result_folder"].values.tolist()
        return summary

    def _match_folder_csv(self, summaryfile, strategy_path):
        correct_file = os.listdir(strategy_path)
        summaryfile = summaryfile[summaryfile["result_folder"].apply(lambda x: x.split("/")[-1] in correct_file)]
        return summaryfile
    
    def _filter_epochs(self, summaryfile):
        # Description
        """
        Filter the summaryfile by epochs and create a dictionary with the path of the experiments
        Args:
        - summaryfile: pd.DataFrame
        - epochs_list: list of int
        Returns:
        - place_holder: Box
        """
        place_holder = Box()
        exp_summary = []
        for e in self.epoch_list:
            summary_e = summaryfile[summaryfile["client.local_epochs"] == e]
            if summary_e.shape[0] > 5:
                summary_e = summary_e.iloc[:-1]
            epochs_path = summary_e["result_folder"].values.tolist()
            place_holder[f'epoch_{e}'] = Box(summary=summary_e, path=epochs_path)
            exp_summary.append(summary_e)
        summary_df = pd.concat(exp_summary).reset_index(drop=True)
        return place_holder, summary_df
    
    def _create_epochs_dict(self, by_epochs):
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
    
    def _create_json_file(self)->Dict[str, Dict[str, Dict[str, Dict]]]:
        strategy_dict = {}
        for strategy in self.strategies:
            print(f"Reading summary file for {strategy}")
            path = os.path.join(self.output_dir, strategy, self.split)
            summary = self.read_summaryfile(path)
            print(f"Summary shape after condition: {summary.shape}")
            by_epochs, summary_epochs = self._filter_epochs(summary)
            strategy_dict.setdefault(strategy, {}).setdefault('exp_summary', summary_epochs.to_dict(orient='records'))
            strategy_dict[strategy]['split_epoch'] = self._create_epochs_dict(by_epochs)
        return strategy_dict
    
    def read_server(self, path_to_result):
        """_summary_
        
        Returns:
            SimpleNamespace[pd.DataFrame]: Contains energy, results as DataFrames
        """
        try :
            results = flwr_pkl(path_to_result)
        except FileNotFoundError as err:
            print(err)
            results = None
            results_df = None
        if results is not None:
            acc_centralized = [acc[1] for acc in results.metrics_centralized["accuracy"][1:]]
            acc_distributed = [acc[1] for acc in results.metrics_distributed["accuracy"]]
            losses_centralized = [loss[1] for loss in results.losses_centralized[1:]] # First loss is evaluated on initial parameters
            losses_distributed = [loss[1] for loss in results.losses_distributed]
            server_round = [i for i in range(1,len(acc_centralized)+1)]
            results_df = pd.DataFrame(
                {
                    "server_round": server_round,
                    "acc_centralized": acc_centralized,
                    "acc_distributed": acc_distributed,
                    "losses_centralized": losses_centralized,
                    "losses_distributed": losses_distributed   
                }
            )
        return results_df
    
    def filter_time_accuracy(self, strategy, epoch, exp)->None:
        """
        Filter time and accuracy from server results based on a threshold.
        Save the information to strategies_dict in each experiments summary.
        e.g. strategies_dict.fedavg.split_epoch.epoch_5.exp_0.summary.result_time
        """
        result_path = self.strategies_dict[strategy]['split_epoch'][epoch][exp]['server']['results']
        log_path = self.strategies_dict[strategy]['split_epoch'][epoch][exp]['server']['logs']
        server_accuracy = self.read_server(result_path)
        metrics = ['acc_centralized', 'acc_distributed']
        results = Box()
        for m in metrics:
            round_threshold = server_accuracy[server_accuracy[m] >= self.threshold]['server_round'].iloc[0]
            acc_threshold = server_accuracy[server_accuracy[m] >= self.threshold][m].iloc[0]
            loss_threshold = server_accuracy[server_accuracy[m] >= self.threshold]['losses_distributed'].iloc[0]
            results[m] = {'round': round_threshold, 'acc': acc_threshold}
        round_acc_cen = results.acc_centralized.round
        round_acc_dis = results.acc_distributed.round
        server_log = server_log_file_to_csv(log_path)
        time_cen = server_log[server_log['round_number'] == round_acc_cen]['timestamp'].iloc[-1]
        time_dis = server_log[server_log['round_number'] == round_acc_dis]['timestamp'].iloc[-1]
        results.acc_centralized['time'] = time_cen
        results.acc_distributed['time'] = time_dis
        strategies_dict[strategy]['split_epoch'][epoch][exp]['summary']['result_time'] = results.to_dict()
        

    def filter_time_accuracy_all(self)->None:
        """
        Filter time and accuracy from server results based on a threshold for all strategies.
        Save the information to strategies_dict in each experiments summary.
        e.g. strategies_dict.fedavg.split_epoch.epoch_5.exp_0.summary.result_time
        """
        for strategy in self.strategies_dict.keys():
            for epoch in self.strategies_dict[strategy]['split_epoch'].keys():
                for exp in self.strategies_dict[strategy]['split_epoch'][epoch].keys():
                    self.filter_time_accuracy(strategy, epoch, exp)

class EnergyResult:
    def __init__(self,strategies_dict:Dict)->None:
        """Read results from path
        Args:
            path_to_output (str): Path to output folder
            nb_clients (int): Nb of clients
            date (str): %Y-%m-%d
            time (str): %H-%M-%S
        """
        self.strategy_dict = strategies_dict
        self.exp_info = summaryfile
        self.path_to_result = summaryfile["result_folder"]
        self.date_time_format = "%Y-%m-%d %H:%M:%S" #.%f
        
    def _folder_still_exist(self):
        return os.path.exists(self.path_to_result)

    def _get_clients_in_host(self) -> List[Tuple[str, List[int]]]:
        """
        Get the client information for each host.

        Returns:
            A list of tuples, where each tuple contains the host name and a list of client IDs.
        """
        hosts = [col for col in self.exp_info.keys() if "estats" in col]
        hostmetainfo = []
        for host in hosts:
            client_list = self.exp_info[host]
            try:
                client_list = re.sub(r'\[|\]', '', client_list).split()
                client_list = [int(i) for i in client_list]
                hostmetainfo.append((host, client_list))
            except TypeError as err:
                print(f"Host {host} doesn't have any clients: {self.exp_info[host]}, triggered {err}")
        return hostmetainfo
    
    def _get_selectedclient_in_host(self) -> List[Tuple[str, List[int]]]:
        """
        Retrieves the selected clients for each host.

        Returns:
            A list of tuples, where each tuple contains the host name and a list of selected clients.
        """
        hosts = sorted([host for host in os.listdir(self.path_to_result) if 'client' in host])
        hostmetainfo = []
        for host in hosts:
            file_in_host = os.listdir(os.path.join(self.path_to_result, host))
            client_list = [name.split("_")[-1].split('.')[0] for name in file_in_host if "evalresult" in name]
            client_list = [int(i) for i in client_list]
            hostmetainfo.append((host, client_list))
        return hostmetainfo
    
    def _match_host_estats(self) -> Dict[str, str]:
        estats = [x for x,_ in self._get_clients_in_host()]
        hosts = [x for x,_ in self._get_selectedclient_in_host()]
        if len(estats) != len(hosts):
            print(estats, hosts)
            raise ValueError("The number of hosts in the summary file does not match the number of hosts in the result folder.")
        estats_match = {}
        for i in range(len(estats)):
            estats_match[hosts[i]] = estats[i]
        
        return estats_match
    
    def _get_client_training_results(self, path_host, cid: int) -> pd.DataFrame:
        """
        Read the evaluation, fit results, and fit times for a specific client.

        Args:
            path_host (str): The path to the host directory.
            cid (int): The client ID.

        Returns:
            Namespace: A namespace object containing the evaluation results, fit results, and fit times.
        """
        evalresult = pd.read_csv(
            os.path.join(path_host, f"evalresult_client_{cid}.csv"), 
            parse_dates=["time"], 
            date_format=self.date_time_format
            )
        fitresult = pd.read_csv(
            os.path.join(path_host, f"fitresult_client_{cid}.csv"), 
            parse_dates=["time"], 
            date_format=self.date_time_format
            )
        fittime = pd.read_csv(
            os.path.join(path_host, f"fittimes_client_{cid}.csv"), 
            parse_dates=["Start Time","End Time"], 
            date_format=self.date_time_format
            )
        # Get fit time of each communication round
        fittime["fittime"] = (fittime["End Time"] - fittime["Start Time"]).dt.total_seconds()
        return SimpleNamespace(results=evalresult, fitresults=fitresult, fittimes=fittime)
        
    def _read_client_host(self, hid:int):
        """
        Reads the client host data for a given host ID.

        Args:
            hid (int): The host ID.

        Returns:
            SimpleNamespace: A namespace object containing the hostname, energy data, and client metadata.
        """
        path_to_host = os.path.join(self.path_to_result,f"client_host_{hid}")
        try:
            energy = pd.read_csv(
                os.path.join(path_to_host,"energy.csv"), 
                parse_dates=["timestamp"]
                )
            energy.columns = [col.strip() for col in energy.columns]
            
            hostmetadata = self._get_selectedclient_in_host()
            try:
                hostname = hostmetadata[hid][0]
                client_list = hostmetadata[hid][1]
                client_data = {f"client_{cid}": self._get_client_training_results(path_to_host,cid) for cid in client_list}
                return SimpleNamespace(hostname=hostname, energy=energy, clients=client_data)
            except IndexError as err:
                print(f"Host {hid} doesn't have any clients: {err}")
        except FileNotFoundError as err:
            print(f"Host {hid} doesn't have an energy file: {err}")
            
    
    def _get_each_client_training_results(self) -> List[pd.DataFrame]:
            """
            Get all the clients' training results from the host and returns a list of pandas DataFrames.

            Returns:
                List[pd.DataFrame]: A list of pandas DataFrames containing the clients' training results.
            """
            clients = []
            hostmetadata = self._get_selectedclient_in_host()
            for hid, _ in zip(range(len(hostmetadata)), hostmetadata):
                hostinfo = self._read_client_host(hid)
                if hostinfo is None:
                    continue
                for k in hostinfo.clients.keys():
                    clients.append(hostinfo.clients[k])            
            return clients
    
    def _get_each_host_energy(self) -> Tuple[List[str], List[pd.DataFrame]]:
        """
        Get the energy information for each hosts.

        Returns:
            A tuple containing two lists:
            - hostname: A list of hostnames.
            - energy: A list of energy dataframes for each host.
        """
        if "estats" in self.exp_info.keys():
            hosts = self._get_clients_in_host()
        else:
            hosts = [x for x in sorted(os.listdir(self.exp_info["result_folder"])) if "client_host" in x]
        hostname, energy = [], []
        for hid in range(len(hosts)):
            hostinfo = self._read_client_host(hid)
            if hostinfo is None:
                continue
            hostname.append(hostinfo.hostname)
            energy.append(hostinfo.energy)
        return hostname, energy

    def _read_server(self)->pd.DataFrame:
        """_summary_

        Returns:
            SimpleNamespace[pd.DataFrame]: Contains energy, results as DataFrames
        """
        path_to_server = os.path.join(self.path_to_result,"server")
        try : 
            energy = pd.read_csv(
                os.path.join(path_to_server,"energy.csv"), 
                parse_dates=["timestamp"]
                )
            # energy["timestamp"] = energy["timestamp"].dt.round("1s") # Rounding to 1s
            energy.columns = [col.strip() for col in energy.columns]
        except FileNotFoundError as err:
            energy = None
            print(err)
        
        try :
            results = flwr_pkl(os.path.join(path_to_server,"results.pkl"))
        except FileNotFoundError as err:
            print(err)
            results = None
            results_df = None
        if results is not None:
            acc_centralized = [acc[1] for acc in results.metrics_centralized["accuracy"][1:]]
            acc_distributed = [acc[1] for acc in results.metrics_distributed["accuracy"]]
            losses_centralized = [loss[1] for loss in results.losses_centralized[1:]] # First loss is evaluated on initial parameters
            losses_distributed = [loss[1] for loss in results.losses_distributed]
            server_round = [i for i in range(1,len(acc_centralized)+1)]
            #print(len(server_round), len(acc_centralized), len(acc_distributed), len(losses_centralized), len(losses_distributed))
            results_df = pd.DataFrame(
                {
                    "server_round": server_round,
                    "acc_centralized": acc_centralized,
                    "acc_distributed": acc_distributed,
                    "losses_centralized": losses_centralized,
                    "losses_distributed": losses_distributed   
                }
            )
        return SimpleNamespace(energy=energy, results=results_df)
        
    def clients_results(self)->List[pd.DataFrame]:
        clients = self._get_each_client_training_results()
        return clients
    
    def server_results(self)->pd.DataFrame:
        server = self._read_server()
        return server
    
    def client_host_energy(self):
        return self._get_each_host_energy()
    

