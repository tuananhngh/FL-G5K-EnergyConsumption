
from typing import Dict, List, Tuple
from types import SimpleNamespace
import pickle as pkl
import re
import os
import yaml # pip install PyYAML
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from flwr.server import strategy
# from torch import mul
from pathlib import Path
from omegaconf import OmegaConf

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



# Function read results from path
def flwr_pkl(path_to_pkl):
    with open(path_to_pkl, "rb") as f:
        result = pkl.load(f)
    return result


def read_yaml_file(path_to_yaml):
    with open(path_to_yaml, "r") as f:
        try : 
            result = yaml.load(f, Loader=yaml.FullLoader)
            result = OmegaConf.create(result)
        except yaml.YAMLError as exc:
            print(exc)
    return result


def read_flwr_logfile(file_path):
    # Read log file
    with open(file_path, 'r') as file:
        log_lines = file.readlines()

    # Define a regex pattern to extract relevant information
    pattern = re.compile(r'\[(.*?)\]\[(.*?)\]\[(.*?)\]\s*-\s*(.*)')

    timestamps = []
    log_sources = []
    message_types = []
    messages = []

    # Process each log line
    for line in log_lines:
        match = pattern.search(line)
        if match:
            timestamp, logsource, logtype ,message = match.groups()
            timestamps.append(timestamp)
            log_sources.append(logsource)
            message_types.append(logtype)
            messages.append(message.strip())

    # Create a DataFrame
    log_data = {'timestamp': timestamps, 'logsource': log_sources, 'logtype': message_types, 'message': messages}
    df = pd.DataFrame(log_data)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f')

    return df

class ReadFlowerLog:
    def __init__(self, path_to_log):
        self.path_to_log = path_to_log
    
    def _log_to_pd(self) -> None:
        with open(self.path_to_log, 'r') as file:
            log_lines = file.readlines()

        # Define a regex pattern to extract relevant information
        pattern = re.compile(r'\[(.*?)\]\[(.*?)\]\[(.*?)\]\s*-\s*(.*)')

        timestamps = []
        log_sources = []
        message_types = []
        messages = []

        # Process each log line
        for line in log_lines:
            match = pattern.search(line)
            if match:
                timestamp, logsource, logtype ,message = match.groups()
                timestamps.append(timestamp)
                log_sources.append(logsource)
                message_types.append(logtype)
                messages.append(message.strip())

        # Create a DataFrame
        log_data = {'timestamp': timestamps, 'logsource': log_sources, 'logtype': message_types, 'message': messages}
        self.df = pd.DataFrame(log_data)

        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f')

    def _get_fit_message(self) -> List[str]:
        self._log_to_pd()
        fit_message = []
        for index,row in self.df.iterrows():
            if "fit_round" in row['message']:
                fit_message.append(index)
        return self.df.iloc[fit_message].reset_index(drop=True)
    
    def get_server_fit_time(self):
        fit_msg = self._get_fit_message()
        concatmsg = pd.concat([fit_msg[::2].reset_index(drop=True),fit_msg[1::2].reset_index(drop=True)], axis=1, keys=['Start', 'End'])
        server_round = [i for i in range(1,len(concatmsg)+1)]
        server_fit_time = pd.DataFrame(
            {"server_round": server_round,
             "starttime": concatmsg.Start.timestamp,
             "endtime": concatmsg.End.timestamp,}
        )
        server_fit_time["fittime"] = (server_fit_time.endtime - server_fit_time.starttime).dt.total_seconds()
        
        return server_fit_time


# log_file_path = '../outputs_from_tl1/2024-01-22_19-31-34/server/main_server.log'
# log_dataframe = ReadFlowerLog(log_file_path)
# msg = log_dataframe._get_fit_message()
# fitmess = log_dataframe.get_server_fit_time()




class EnergyResult:
    def __init__(self,path_to_result:str, summaryfile:pd.DataFrame)->None:
        """Read results from path
        Args:
            path_to_output (str): Path to output folder
            nb_clients (int): Nb of clients
            date (str): %Y-%m-%d
            time (str): %H-%M-%S
        """
        self.summaryfile = summaryfile
        self.path_to_result = path_to_result
        self.exp_info = self.summaryfile[self.summaryfile["result_folder"]==self.path_to_result]
        #print(self.exp_info)
        self.date_time_format = "%Y-%m-%d %H:%M:%S"

        
        

    def _get_client_in_host(self)->List[Tuple[str,List[int]]]:
        #exp_info = self.summary[self.summary["result_folder"]==self.result_path]
        hosts = [col for col in self.exp_info.columns if "estats" in col]
        hostmetainfo = []
        for host in hosts:
            client_list = self.exp_info[host].iloc[0]
            client_list = re.sub(r'\[|\]','',client_list).split()
            client_list = [int(i) for i in client_list]
            hostmetainfo.append((host,client_list))
        return hostmetainfo
    
    def _get_selectedclient_in_host(self)->List[Tuple[str,List[int]]]:
        hosts = sorted([host for host in os.listdir(self.path_to_result) if 'client' in host])
        hostmetainfo = []
        for host in hosts:
            file_in_host = os.listdir(os.path.join(self.path_to_result,host))
            client_list = [name.split("_")[-1].split('.')[0] for name in file_in_host if "evalresult" in name]
            client_list = [int(i) for i in client_list]
            hostmetainfo.append((host,client_list))
        return hostmetainfo
        
    def _read_client_host(self, hid:int):
        path_to_host = os.path.join(self.path_to_result,f"client_host_{hid}")
        energy = pd.read_csv(os.path.join(path_to_host,"energy.csv"), parse_dates=["timestamp"])
        energy.columns = [col.strip() for col in energy.columns]
        
        #hostmetadata = self._get_client_in_host()
        hostmetadata = self._get_selectedclient_in_host()
        hostname = hostmetadata[hid][0]
        client_list = hostmetadata[hid][1]
        client_metadata = {f"client_{cid}": self._read_client(path_to_host,cid) for cid in client_list}
        #client_metadata = SimpleNamespace(**client_metadata)
        return SimpleNamespace(hostname=hostname, energy=energy, clients=client_metadata)    
    
        
    def _read_client(self,path_host,cid:int)->pd.DataFrame:
        evalresult = pd.read_csv(os.path.join(path_host, f"evalresult_client_{cid}.csv"), parse_dates=["time"], date_format=self.date_time_format)
        
        fitresult = pd.read_csv(os.path.join(path_host, f"fitresult_client_{cid}.csv"), parse_dates=["time"], date_format=self.date_time_format)
        
        fittime = pd.read_csv(os.path.join(path_host, f"fittimes_client_{cid}.csv"), parse_dates=["Start Time","End Time"], date_format=self.date_time_format)
        fittime["fittime"] = (fittime["End Time"] - fittime["Start Time"]).dt.total_seconds() # Get fit time of each communication round
        return SimpleNamespace(results=evalresult, fitresults=fitresult, fittimes=fittime)
    
    def _read_all_clients(self)->List[pd.DataFrame]:
        clients = []
        hostmetadata = self._get_client_in_host()
        for hid,hostdata in zip(range(len(hostmetadata)),hostmetadata):
            hostinfo = self._read_client_host(hid)
            for k in hostinfo.clients.keys():
                clients.append(hostinfo.clients[k])            
        return clients
    
    def _read_all_host_energy(self)->Tuple[List[str],List[pd.DataFrame]]:
        hosts = self._get_client_in_host()
        hostname, energy = [], []
        for hid in range(len(hosts)):
            hostinfo = self._read_client_host(hid)
            hostname.append(hostinfo.hostname)
            energy.append(hostinfo.energy)
        return hostname, energy

    def _read_server(self)->pd.DataFrame:
        """_summary_

        Returns:
            SimpleNamespace[pd.DataFrame]: Contains energy, results as DataFrames
        """
        path_to_server = os.path.join(self.path_to_result,"server")
        energy = pd.read_csv(os.path.join(path_to_server,"energy.csv"), parse_dates=["timestamp"])
        energy["timestamp"] = energy["timestamp"].dt.round("1s") # Rounding to 1s
        energy.columns = [col.strip() for col in energy.columns]
        
        results = flwr_pkl(os.path.join(path_to_server,"results.pkl"))
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
    
    def _filter_energy(self, df:pd.DataFrame, start_time:str, end_time:str)->pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            start_time (str): _description_
            end_time (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
        return df
    
        
    def clients_results(self)->List[pd.DataFrame]:
        clients = self._read_all_clients()
        return clients
    
    def server_results(self)->pd.DataFrame:
        server = self._read_server()
        return server
    
    def client_host_energy(self):
        hostname, energy = self._read_all_host_energy()
        return hostname, energy
            
    def make_energy_plot(self, attribute:str, columns_name_1:str, columns_name_2)->None:
        """_summary_

        Args:
            attribute (str): Output of client or server results. Attribute of SimpleNamespace
            columns_name_1 (str): Columns name of DataFrame for the x-axis
            columns_name_2 (_type_): Columns name of DataFrame for the y-axis
            Example: attribute = "energy", columns_name_1 = "timestamp", columns_name_2 = "tot inst power"
        """
        # clients = self._read_all_clients()
        server = self._read_server()
        hostname, host_energy = self.client_host_energy()
        plt.figure(figsize=(10,5))
        for hid in range(len(host_energy)):
        #    plt.plot(host[cid].__getattribute__(attribute)[columns_name_1], clients[cid].__getattribute__(attribute)[columns_name_2], label=f"Client {cid}")
            plt.plot(host_energy[hid][columns_name_1], host_energy[hid][columns_name_2], label=f"{hostname[hid]}")
        plt.plot(server.__getattribute__(attribute)[columns_name_1], server.__getattribute__(attribute)[columns_name_2], label="Server")
        plt.xlabel(columns_name_1)
        plt.ylabel(columns_name_2)
        plt.legend()
        plt.show()
        
    def make_server_plot(self,config, centralized:bool=True,**kwargs)->None:
        """
        Args:
            attribute (str): _description_
        """
        config_text = '\n'.join(f'{k}: {v}' for k, v in config.items())
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        server = self._read_server()
        #clients= self._read_all_clients()
        for k,v in kwargs.items():
            if k=="loss":
                fig,ax = plt.subplots(figsize=(10,5))
                # for cid in range(len(clients)):
                #    plt.plot(clients[cid].__getattribute__(v[0])[v[1]],clients[cid].__getattribute__(v[0])[v[2]], label=f"Client {cid}")
                if centralized:
                    x_vals = server.__getattribute__(v[0])[v[1]]
                    y_vals = server.__getattribute__(v[0])[v[3]]
                    min_value = min(y_vals)
                    ax.hlines(y=min_value, label=f"Min {v[3]} : {min_value:.4f}", linestyle="--", xmin = x_vals.min(), xmax = x_vals.max(), color="blue")
                    ax.plot(x_vals, y_vals, label=f"Server {v[3]}", linestyle="--")
                ax.text(1.05, 0.95, config_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                
                xd_vals = server.__getattribute__(v[0])[v[1]]
                yd_vals = server.__getattribute__(v[0])[v[4]]
                mind_value = min(yd_vals)
                avg_val = np.sum(yd_vals)/len(yd_vals)
                ax.hlines(y=mind_value, label=f"Min & Avg {v[4]} : {mind_value:.4f} & {avg_val:.4f}", linestyle="--", color="orange", xmin =xd_vals.min(), xmax = xd_vals.max())
                ax.plot(xd_vals, yd_vals, label=f"Server {v[4]}", linestyle="-.")
                ax.set_xlabel(v[1])
                ax.set_ylabel(v[2])
                ax.legend()

            elif k=="accuracy":
                fig, ax = plt.subplots(figsize=(10,5))
                #plt.figure(figsize=(10,5))
                # for cid in range(len(clients)):
                #    plt.plot(clients[cid].__getattribute__(v[0])[v[1]],clients[cid].__getattribute__(v[0])[v[2]], label=f"Client {cid}")
                if centralized:
                    x_vals = server.__getattribute__(v[0])[v[1]]
                    y_vals = server.__getattribute__(v[0])[v[3]]
                    max_value = max(y_vals)
                    ax.hlines(y=max_value, label=f"Max {v[3]} : {max_value:.4f}", linestyle="--", xmin = x_vals.min(), xmax = x_vals.max(), color="blue")
                    ax.plot(x_vals, y_vals, label=f"Server {v[3]}", linestyle="--")
                ax.text(1.05, 0.95, config_text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                
                xd_vals = server.__getattribute__(v[0])[v[1]]
                yd_vals = server.__getattribute__(v[0])[v[4]]
                maxd_value = max(yd_vals)
                avg_val = np.sum(yd_vals)/len(yd_vals)
                ax.hlines(y=maxd_value, label=f"Max & Avg {v[4]} : {maxd_value:.4f} & {avg_val:.4f}", linestyle="--", color="orange", xmin =xd_vals.min(), xmax = xd_vals.max())
                ax.plot(xd_vals,yd_vals, label=f"Server {v[4]}", linestyle="-.")
                ax.set_xlabel(v[1])
                ax.set_ylabel(v[2])
                plt.legend()
                

    
def match_folder_csv(summaryfile, output_dir):
    correct_file = os.listdir(output_dir)
    summaryfile = summaryfile[summaryfile["result_folder"].apply(lambda x: x.split("/")[-1] in correct_file)]
    return summaryfile

def select_model(summaryfile, model_name):
    summaryfile = summaryfile[summaryfile["neuralnet"]==model_name]
    return summaryfile
    
def read_summaryfile(path_summary):
    usr_home = path_summary.replace('/',' ').split()
    usr_homedir = f"{usr_home[0]}/{usr_home[1]}"
    summary = pd.read_csv(path_summary)
    cols = summary.columns
    summary["result_folder"] = summary["result_folder"].apply(lambda x: x.replace("root",usr_homedir))
    return summary

def config_drop(config:Dict[str,str])->Dict[str,str]:
    keys_etc = ['server','energy_file','sleep_duration']
    
    nodes_keys = [k for k in config if 'estats' in k]
    timestamp_keys = [k for k in config if 'timestamp' in k]
    comm_keys = [k for k in config if 'comm' in k]
    params_keys = [k for k in config if 'params' in k]
    additional_keys = [k for k in config for i in keys_etc if i in k]
    
    results_dir_short = config["result_folder"].split("/")[-1]
    config["result_folder"] = results_dir_short
    
    drop_list = nodes_keys + timestamp_keys + comm_keys + additional_keys #+ params_keys
    for key in drop_list:
        config.pop(key)
    return config
        

if __name__ == "__main__":  
    result_plot = {#"loss": ["results","server_round","loss","losses_centralized","losses_distributed"],}
                    "accuracy": ["results","server_round","accuracy","acc_centralized","acc_distributed"]}
    
    path_to_output = "/home/tunguyen/energyfl/outputs3/"
    summary_path = os.path.join(path_to_output,"experiment_summary.csv")
    summaryfile = read_summaryfile(summary_path)
    summaryfile = match_folder_csv(summaryfile, path_to_output)
    summaryfile = select_model(summaryfile, "Net") # select only ResNet
    results_dir_ls = summaryfile["result_folder"].tolist()
    summaryfile_dict = summaryfile.to_dict(orient="records")
    for (result_dir,config) in zip(results_dir_ls,summaryfile_dict):
        result = EnergyResult(result_dir,summaryfile)
        #result.make_energy_plot("energy",'timestamp',"tot avg power (mW)")
        config = config_drop(config)
        result.make_server_plot(config,centralized=True, **result_plot)
        
