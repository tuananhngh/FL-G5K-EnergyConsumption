from typing import Dict, List
from types import SimpleNamespace
import pickle as pkl
# import os
import yaml # pip install PyYAML
import matplotlib.pyplot as plt
import pandas as pd
# from flwr.server import strategy
# from torch import mul
from pathlib import Path
from omegaconf import OmegaConf

energy_cols = ['timestamp',
 'RAM%',
 'GPU%',
 'GPU inst power',
 'GPU avg power',
 'CPU%',
 'CPU inst power',
 'CPU avg power',
 'tot inst power',
 'tot avg power']

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

#pathyaml = Path("../outputs_from_tl/main_server_0/2024-01-20/23-35-19/.hydra/config.yaml")
#ok = read_yaml_file(pathyaml)

class EnergyResult:
    def __init__(self, path_to_output:str, nb_clients:int, datetime:str)->None:
        """Read results from path
        Args:
            path_to_output (str): Path to output folder
            nb_clients (int): Nb of clients
            date (str): %Y-%m-%d
            time (str): %H-%M-%S
        """
        self.path_to_output = Path(path_to_output)
        self.nb_clients = nb_clients
        self.datetime = datetime
        self.cids = [i for i in range(nb_clients)]
        self.date_time_format = "%Y-%m-%d %H:%M:%S"
        
    def _read_client(self, cid:int)->pd.DataFrame:
        """_summary_

        Args:
            cid (int): _description_

        Returns:
            SimpleNamespace: Contains energy, evalresult, fitresult, fittimes as DataFrames
            To access an element: SimpleNamespace.element
        """
        path_to_client = self.path_to_output/self.datetime/f"client_{cid}"
        
        energy = pd.read_csv(path_to_client/"energy.csv", parse_dates=["timestamp"])
        # energy["timestamp"] = energy["timestamp"].dt.round("1s") # Rounding to 1s
        energy.columns = [col.strip() for col in energy.columns]
        
        evalresult = pd.read_csv(path_to_client/"evalresult.csv", parse_dates=["time"], date_format=self.date_time_format)
        fitresult = pd.read_csv(path_to_client/"fitresult.csv", parse_dates=["time"], date_format=self.date_time_format)
        fittimes = pd.read_csv(path_to_client/"fittimes.csv", parse_dates=["Start Time","End Time"], date_format=self.date_time_format)

        fittimes["fittime"] = (fittimes["End Time"] - fittimes["Start Time"]).dt.total_seconds() # Get fit time of each communication round
        return SimpleNamespace(energy=energy, results=evalresult, fitresult=fitresult, fittimes=fittimes)
    
        
    def _read_server(self)->pd.DataFrame:
        """_summary_

        Returns:
            SimpleNamespace[pd.DataFrame]: Contains energy, results as DataFrames
        """
        path_to_server = self.path_to_output/self.datatime/"server"
        energy = pd.read_csv(path_to_server/"energy.csv", parse_dates=["timestamp"])
        energy["timestamp"] = energy["timestamp"].dt.round("1s") # Rounding to 1s
        energy.columns = [col.strip() for col in energy.columns]
        
        results = flwr_pkl(path_to_server/"results.pkl")
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
        
    def _read_all_clients(self)->List[pd.DataFrame]:
        clients = []
        for cid in self.cids:
            clients.append(self._read_client(cid))
        return clients
    
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
        
    def make_energy_plot(self, attribute:str, columns_name_1:str, columns_name_2)->None:
        """_summary_

        Args:
            attribute (str): Output of client or server results. Attribute of SimpleNamespace
            columns_name_1 (str): Columns name of DataFrame for the x-axis
            columns_name_2 (_type_): Columns name of DataFrame for the y-axis
            Example: attribute = "energy", columns_name_1 = "timestamp", columns_name_2 = "tot inst power"
        """
        clients = self._read_all_clients()
        server = self._read_server()
        plt.figure(figsize=(10,5))
        for cid in self.cids:
            plt.plot(clients[cid].__getattribute__(attribute)[columns_name_1], clients[cid].__getattribute__(attribute)[columns_name_2], label=f"Client {cid}")
        plt.plot(server.__getattribute__(attribute)[columns_name_1], server.__getattribute__(attribute)[columns_name_2], label="Server")
        plt.xlabel(columns_name_1)
        plt.ylabel(columns_name_2)
        plt.legend()
        plt.show()
        
    def make_result_plot(self,**kwargs)->None:
        """
        Args:
            attribute (str): _description_
        """
        server = self._read_server()
        clients= self._read_all_clients()
        for k,v in kwargs.items():
            if k=="loss":
                plt.figure(figsize=(10,5))
                for cid in self.cids:
                    plt.plot(clients[cid].__getattribute__(v[0])[v[1]],clients[cid].__getattribute__(v[0])[v[2]], label=f"Client {cid}")
                plt.plot(server.__getattribute__(v[0])[v[1]],server.__getattribute__(v[0])[v[3]], label=f"Server {v[3]}", linestyle="--")
                plt.plot(server.__getattribute__(v[0])[v[1]],server.__getattribute__(v[0])[v[4]], label=f"Server {v[4]}", linestyle="-.")
                plt.xlabel(v[1])
                plt.ylabel(v[2])
                plt.legend()
            elif k=="accuracy":
                plt.figure(figsize=(10,5))
                for cid in self.cids:
                    plt.plot(clients[cid].__getattribute__(v[0])[v[1]],clients[cid].__getattribute__(v[0])[v[2]], label=f"Client {cid}")
                plt.plot(server.__getattribute__(v[0])[v[1]],server.__getattribute__(v[0])[v[3]], label=f"Server {v[3]}", linestyle="--")
                plt.plot(server.__getattribute__(v[0])[v[1]],server.__getattribute__(v[0])[v[4]], label=f"Server {v[4]}", linestyle="-.")
                plt.xlabel(v[1])
                plt.ylabel(v[2])
                plt.legend()
                    
        

if __name__ == "__main__":  
    
    result_plot = {"loss": ["results","server_round","loss","losses_centralized","losses_distributed"],
                    "accuracy": ["results","server_round","accuracy","acc_centralized","acc_distributed"]}
        
    result = EnergyResult("./outputs/", 6, "2024-01-22_19-31-34")
    mycsv = result._read_client(5)
    server = result._read_server()
    result.make_energy_plot("energy",'timestamp',"tot inst power")
    result.make_result_plot(**result_plot)

# def read_result(path_to_last_run, multirun=True):
#     ls_results = {}
#     if multirun:
#         run_dirs = os.listdir(path_to_last_run)
#         for rd in run_dirs:
#             run_path = os.path.join(path_to_last_run, rd)
#             if os.path.isdir(run_path):
#                 if os.path.exists(os.path.join(run_path, "results.pkl")):
#                     pkl_file = os.path.join(run_path, "results.pkl")
#                     result = read_pkl(pkl_file)
#                     ls_results[rd] = result
#     else:
#         pkl_file = os.path.join(path_to_last_run, "results.pkl")
#         result = read_pkl(pkl_file)
#         ls_results["0"] = result
#     return ls_results   

# def plot_results_multirun(result_file:Dict, metrics)->None:
#     #nb_runs = len(result_file)
#     #Plot run results on the same figure
#     fig,ax = plt.subplots(figsize=(5,5))
#     labels = {'0': 'FedOrdered',
#               '1': 'FedAvg'}
#     for k,v in result_file.items():
#         if metrics == "losses_distributed":
#             rounds, vals = zip(*v.losses_distributed)
#         elif metrics == "losses_centralized":
#             rounds, vals = zip(*v.losses_centralized)
#         elif metrics == "metrics_centralized":
#             rounds, vals = zip(*v.metrics_centralized["accuracy"])
#         elif metrics == "metrics_distributed":
#             rounds, vals = zip(*v.metrics_distributed["accuracy"])
#         ax.plot(rounds,vals, label=labels[k])
#     ax.set_xlabel('Communication Round')
#     ax.set_ylabel(metrics)
#     ax.legend()
    
    

# path_to_multirun = "/Users/Slaton/Documents/grenoble-code/fl-flower/outputs_from_tl/main_server_0/2024-01-20/23-35-19"
# ok = read_result(path_to_multirun, multirun=False)
# plot_results_multirun(ok, "losses_distributed")

