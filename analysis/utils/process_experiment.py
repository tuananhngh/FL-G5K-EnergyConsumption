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
from .read_file_functions import (
    server_log_file_to_csv, 
    load_client_data, 
    load_server_data,
)
from .process_data_functions import read_server

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


class ProcessResults:
    def __init__(self, 
                 *,
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
        if self.threshold is not None:
            self.filter_time_accuracy_all()
        
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
        #print(f"Read Summary {summary.shape}")
        summary = self._match_folder_csv(summary, strategy_path)
        #print(f"Summary shape before matching condition : {summary.shape}")
        # Filter by parameters
        summary["result_folder"] = summary["result_folder"].apply(lambda x: x.replace("/root",  self.usr_homedir))
        if self.condition is not None: 
            summary = summary.loc[self.condition(summary)]
        #print(f"Summary shape after condition: {summary.shape}")
        # Filter by subfolder
        #folder_path = summary["result_folder"].values.tolist()
        return summary

    def _match_folder_csv(self, summaryfile, strategy_path):
        correct_file = os.listdir(strategy_path)
        summaryfile = summaryfile[summaryfile["result_folder"].apply(lambda x: x.split("/")[-1] in correct_file)]
        return summaryfile
    
    def summary_file(self, strategy):
        summary_pd = pd.DataFrame(self.strategies_dict[strategy]['exp_summary'])
        return summary_pd
    
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
                    host_name = subfolder[k][0]
                    host_name = host_name.replace('client_host','host')
                    host_path = subfolder[k][1]
                    if os.path.isdir(host_path):
                        files = os.listdir(host_path)
                        for e,file in enumerate(files):
                            if file == 'client.log' :
                                files[e] = 'client_log'
                            elif file == 'server.log':
                                files[e] = 'server_log'
                            elif file == 'client_pids.csv':
                                files[e] = 'client_pid'
                            else:
                                #print(re.split('[.]', file)[0])
                                files[e] = re.split('[.]', file)[0]
                        result_files = [(name, os.path.join(host_path,file)) for name,file in zip(files,os.listdir(host_path))]
                        for file_name, file_path in result_files:
                            #print(f'exp_{i} {host_name} {file_name} {file_path}')
                            byhost[f'exp_{i}'].setdefault(host_name, {}).setdefault(file_name,file_path)
                        #byhost[f'exp_{i}'][client_name] = subfolder[k][1]
            #print(f'by host {byhost}')
            epochs_dict[epoch] = byhost
        return epochs_dict
    
    def _create_json_file(self)->Dict[str, Dict[str, Dict[str, Dict]]]:
        strategy_dict = {}
        for strategy in self.strategies:
            print(f"Reading summary file for {strategy}")
            path = os.path.join(self.output_dir, strategy, self.split)
            summary = self.read_summaryfile(path)
            by_epochs, summary_epochs = self._filter_epochs(summary)
            strategy_dict.setdefault(strategy, {}).setdefault('exp_summary', summary_epochs.to_dict(orient='records'))
            strategy_dict[strategy]['split_epoch'] = self._create_epochs_dict(by_epochs)
        return strategy_dict
    
    
    def filter_time_accuracy(self, strategy, epoch, exp)->None:
        """
        Filter time and accuracy from server results based on a threshold.
        Save the information to strategies_dict in each experiments summary.
        e.g. strategies_dict.fedavg.split_epoch.epoch_5.exp_0.summary.result_time
        """
        result_path = self.strategies_dict[strategy]['split_epoch'][epoch][exp]['server']['results']
        log_path = self.strategies_dict[strategy]['split_epoch'][epoch][exp]['server']['logs']
        server_accuracy = read_server(result_path)
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
        self.strategies_dict[strategy]['split_epoch'][epoch][exp]['summary']['result_time'] = results.to_dict()
        

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
                    
    def match_expsummary_to_index(self, strategy):
        placeholder = {}
        exp_summary_df = self.summary_file(strategy)
        epochs = exp_summary_df['client.local_epochs'].unique()
        for e in epochs:
            exp_info = exp_summary_df[exp_summary_df['client.local_epochs'] == e]
            print(f"Exp info : {exp_info.index}")
            exp_lists = self.strategies_dict[strategy]['split_epoch'][f'epoch_{e}']
            exp_idx_list = sorted(exp_lists.keys())
            print(f"Exp idx list : {exp_idx_list}")
            output_files = [exp_lists[exp]['summary']['result_folder'] for exp in exp_idx_list]
            # Find index of the experiment in the summary file
            for exp_idx, exp_path in zip(exp_idx_list,output_files):
                exp_idx_summary = exp_info[exp_info['result_folder'] == exp_path].index[0]
                placeholder[exp_path]={'epoch':f'epoch_{e}', 'exp':exp_idx, 'summary_idx':exp_idx_summary}            
        return placeholder

        
class EnergyResults(ProcessResults):
    def __init__(self, 
                 *,
                 output_dir:str, 
                 usr_homedir:str, 
                 strategies:List[str],
                 condition:Callable=None, 
                 threshold:Optional[float]=0.75,
                 epoch_list:List[int] = [1,5],
                 split='labelskew'):
        super().__init__(output_dir=output_dir, 
                         usr_homedir = usr_homedir, 
                         strategies=strategies, 
                         condition=condition, 
                         threshold=threshold, 
                         epoch_list=epoch_list, 
                         split=split)
    
    def match_file_in_dict(self, strategy, epoch:str, exp:str):
        return self.strategies_dict[strategy]['split_epoch'][epoch][exp]
    
    def _get_selectedclient_in_host(self, strategy:str, epoch:str, exp:str)->List[Tuple[str, List[int]]]:
        """
        Retrieves the selected clients for each host.
        Args:
            strategy: The strategy name. e.g fedavg
            epoch: The epoch name. e.g epoch_1
            exp: The experiment name. e.g exp_0
        Returns:
            A list of tuples, where each tuple contains the host name and a list of selected clients.
        """
        exp_files = self.strategies_dict[strategy]['split_epoch'][epoch][exp]
        hosts = sorted([key for key in exp_files.keys() if 'host' in key])
        hostmetainfo = []
        for host in hosts:
            host_files = self.strategies_dict[strategy]['split_epoch'][epoch][exp][host]
            file_in_host  = [name for name in host_files.keys()]
            client_list = [name.split("_")[-1].split('.')[0] for name in file_in_host if "evalresult" in name]
            client_list = sorted([int(i) for i in client_list])
            hostmetainfo.append((host, client_list))
        return hostmetainfo

    def _get_clients_in_host(self, strategy:str, epoch:str, exp:str) -> List[Tuple[str, List[int]]]:
        """
        Get the client information for each host.
        Args:
            strategy: The strategy name. e.g fedavg
            epoch: The epoch name. e.g epoch_1
            exp: The experiment name. e.g exp_0
        Returns:
            A list of tuples, where each tuple contains the host name and a list of client IDs.
        """
        exp_path = self.get_experiment_summary(strategy, epoch, exp)['result_folder']
        strategy_summary = self.strategies_dict[strategy]['exp_summary']
        exp_summary = [exp for exp in strategy_summary if exp['result_folder'] == exp_path][0]
        has_estats = any('estats' in col for col in exp_summary.keys())
        if not has_estats:
            print('No estats in summary file'.upper())
            hostmetainfo = self._get_selectedclient_in_host(strategy, epoch, exp)
            return hostmetainfo
        else:
            hostmetainfo = []
            hosts = [col for col in exp_summary.keys() if "estats" in col]
            print(f"Hosts in summary file : {hosts}")
            for host in hosts:
                client_list = exp_summary[host]
                try:
                    client_list = re.sub(r'\[|\]', '', client_list).split()
                    client_list = [int(i) for i in client_list]
                    hostmetainfo.append((host, client_list))
                except TypeError as err:
                    print(f"Host {host} doesn't have any clients: {self.exp_info[host]}, triggered {err}")
            return hostmetainfo
    
    def _match_host_estats(self, strategy, epoch, exp) -> Dict[str, str]:
        estats = [x for x,_ in self._get_clients_in_host(strategy, epoch, exp)]
        hosts = [x for x,_ in self._get_selectedclient_in_host(strategy, epoch, exp)]
        if len(estats) != len(hosts):
            print(estats, hosts)
            raise ValueError("The number of hosts in the summary file does not match the number of hosts in the result folder.")
        estats_match = {}
        for i in range(len(estats)):
            estats_match[hosts[i]] = estats[i]
        return estats_match

    
    def _get_client_training_results(self, host_files:dict, cid: int) -> Dict[str, pd.DataFrame]:
        """
        Read the evaluation, fit results, and fit times for a specific client.

        Args:
            path_host (str): The path to the host directory.
            cid (int): The client ID.

        Returns:
            Namespace: A namespace object containing the evaluation results, fit results, and fit times.
        """
        evalresult = host_files[f"evalresult_client_{cid}"]
        fitresult = host_files[f"fitresult_client_{cid}"]
        fittime = host_files[f"fittimes_client_{cid}"]
        # Get fit time of each communication round
        fittime["fittime"] = (fittime["end time"] - fittime["start time"]).dt.total_seconds()
        results = {'results':evalresult, 'fitresults':fitresult, 'fittimes':fittime}
        return Box(results)
    
    def _read_client_host(self, strategy:str, epoch:str, exp:str, hid:int):
        """
        Read clients host data for a given host ID.

        Args:
            hid (int): The host ID.

        Returns:
            SimpleNamespace: A namespace object containing the hostname, energy data, and client metadata.
        """
        if self.threshold is not None:
            host_files = load_client_data(self.strategies_dict, strategy, epoch, exp, hid, filter=True)
        else:
            host_files = load_client_data(self.strategies_dict, strategy, epoch, exp, hid, filter=False)
        files_name = [name for name in host_files.keys()]
        #print(f"Files in host {hid} : {sorted(files_name)}")
        hostmetadata = self._get_selectedclient_in_host(strategy, epoch, exp)
        hostname = hostmetadata[hid][0]
        client_list = hostmetadata[hid][1]
        #print(f"Client list : {client_list}")
        results = {}
        try:
            for file in host_files:
                #print(f"File {file} in host {hostname}".upper())
                results[file] = host_files[file]
            client_data = {f"client_{cid}":self._get_client_training_results(host_files, cid) for cid in client_list}
            results['hostname'] = hostname
            results['clients'] = client_data
        except KeyError as e:
            print(f"Host {hostname} doesn't have data: {e}")
        return Box(results) 
    
    def _get_all_client_training_results(self,strategy:str, epoch:str, exp:str) -> List[pd.DataFrame]:
        """
        Get all the clients' training results from the host and returns a list of pandas DataFrames.

        Returns:
            List[pd.DataFrame]: A list of pandas DataFrames containing the clients' training results.
        """
        clients = []
        hostmetadata = self._get_selectedclient_in_host(strategy, epoch, exp)
        #print(f"Host metadata : {hostmetadata}")
        for hid, _ in zip(range(len(hostmetadata)), hostmetadata):
            try :
                hostinfo = self._read_client_host(strategy, epoch, exp, hid)
                for k in hostinfo.clients.keys():
                    clients.append(hostinfo.clients[k])
            except Exception as e:
                result_folder = self.match_file_in_dict(strategy, epoch, exp)['summary']['result_folder'].split('/')[-1]
                print(f"Error : Strategy {strategy}, Folder {result_folder}, Host {hid} doesn't have : {e}".upper())              
        return clients
    
    def _get_all_host_energy(self, strategy:str, epoch:str, exp:str) -> Tuple[List[str], List[pd.DataFrame]]:
        """
        Get the energy information for each hosts.

        Returns:
            A tuple containing two lists:
            - hostname: A list of hostnames.
            - energy: A list of energy dataframes for each host.
        """
        hosts = self._get_clients_in_host(strategy, epoch, exp)
        hostname, energy = [], []
        for hid in range(len(hosts)):
            hostinfo = self._read_client_host(strategy, epoch, exp, hid)
            if hostinfo is None:
                continue
            try:
                #print(f"Host {hostinfo.hostname}".upper())
                energy.append(hostinfo.energy)
                hostname.append(hostinfo.hostname)
            except KeyError as e:
                result_folder = self.get_experiment_summary(strategy, epoch, exp)['result_folder']
                print(f"Strategy :{strategy}, Epoch {epoch} exp {exp} Host {hosts[hid]} Error: {e}")
        return hostname, energy
    
    def _get_server_results(self, strategy:str, epoch:str, exp:str) -> Dict[str, pd.DataFrame]:
        """
        Get the server results for a given strategy, epoch, and experiment.

        Args:
            strategy (str): The strategy name.
            epoch (int): The epoch number.
            exp (int): The experiment number.

        Returns:
            Dict: A dictionnary of pandas DataFrame containing the server results.
        """
        if self.threshold is not None:
            server_files = load_server_data(self.strategies_dict, strategy, epoch, exp, filter=True)
        else:
            server_files = load_server_data(self.strategies_dict, strategy, epoch, exp, filter=False)
        exp_path = self.get_experiment_summary(strategy, epoch, exp)['result_folder']
        results_path = os.path.join(exp_path, 'server', 'results.pkl')
        server_files['results'] = read_server(results_path)
        return server_files
    
    def get_experiment_summary(self, strategy:str, epoch:str, exp:str)->Dict[str, Dict]:
        """
        Fetch the experiment summary for a given strategy, epoch, and experiment.
        """
        return self.strategies_dict[strategy]['split_epoch'][epoch][exp]['summary']
    
    def get_clients_in_host(self, strategy:str, epoch:str, exp:str)->List[Tuple[str, List[int]]]:
        return self._get_clients_in_host(strategy, epoch, exp)
        
    def get_clients_results(self,strategy:str, epoch:str, exp:str)->List[pd.DataFrame]:
        clients = self._get_all_client_training_results(strategy, epoch, exp)
        return clients
    
    def get_server_results(self, strategy:str, epoch:str, exp:str)->Dict[str,pd.DataFrame]:
        server = self._get_server_results(strategy, epoch, exp)
        return server
    
    def get_client_host_energy(self, strategy:str, epoch:str, exp:str)->Tuple[List[str], List[pd.DataFrame]]:
        return self._get_all_host_energy(strategy, epoch, exp)
    
    def get_perf_filtered_info(self, strategy:str, epoch:str, exp:str)->Dict[str, pd.DataFrame]:
        """
        Get the filtered perf information for a given strategy, epoch, and experiment.
        """
        exp_info = self.get_experiment_summary(strategy, epoch, exp)
        result_time = exp_info['result_time']
        infos = {}
        for key in result_time.keys():
            if key == 'acc_centralized' or key == 'acc_distributed':
                infos[f"filtered_{key}_round"] = result_time[key]['round']
                infos[f"filtered_{key}_time"] = result_time[key]['time']
                infos[f"filtered_{key}"] = result_time[key]['acc']
        return infos
    
    def get_clients_host_data(self, strategy:str, epoch:str, exp:str, hid:int):
        """
        Get the clients data for a given host ID.
        """
        return self._read_client_host(strategy, epoch, exp, hid)