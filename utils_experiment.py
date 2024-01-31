
from types import SimpleNamespace
import warnings
from execo import SshProcess, Remote
from execo.host import Host
from execo_g5k import get_oar_job_nodes
from execo_engine import Engine, logger, ParamSweeper, sweep
from datetime import datetime
from typing import Any, Dict, Tuple, List
from pathlib import Path
from box import Box
from pytest import Instance
from experiments import execute_command_on_server_and_clients, concat_dict
import pandas as pd
import yaml
import os
import logging
import time
import shutil


logger = logging.getLogger("MyEXP")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_host_ip(hostname):
    command = f"hostname -I"
    process = SshProcess(command, host=hostname)
    process.run()
    if process.ok:
        ip_address = process.stdout.strip()
        process.kill()
        return ip_address
    else:
        process.kill()
        return f"Failed to get IP for {hostname}"


class MyExperiment(Engine):
    def __init__(self, params:Dict, nodes:List[Host], **kwargs):
        super(MyExperiment, self).__init__()
        self.sweep_name = "sweep"
        self.params = params
        self.nodes = nodes
        self.server = nodes[0]
        self.clients = nodes[1:]
        self.repository_dir = os.getcwd() #TO DECIDE WHERE TO PUT
        self.tmp_dir = "/tmp"
        self.output_name = "outputs"
        
        self.extra_args = Box(kwargs)
        logger.info(("Server : {} \n Clients: {}".format(self.server, self.clients)))
        
    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setattr__(__name, __value)
    
    def __getattr__(self, __name: str) -> Any:
        return super().__getattr__(__name)
    
    
    def _get_kwargs(self, name):
        if name not in self.extra_args:
            self.extra_args.name = None
            warnings.warn(f"Argument {name} not defined and set to None")
        return self.extra_args.name    
    
    def experiment_summary(self)->None:
        self.jetson_sensor_monitor_file = os.path.join(self.repository_dir,"jetson_monitoring_energy.py")
        self.exp_csv_file = os.path.join(self.repository_dir, self.output_name, "experiment_summary.csv")
    
    def _run_dir_setup(self):
        """
        Define experiment runtime hyperparameters
        Args: None
        Returns:
            SimpleNamespace:result_folder: path to result folder
                            tmp_result_folder: path to tmp result folder
                            exp_datetime: datetime of the experiment
                            energy_file: name of the energy file
        """
        exp_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_folder = os.path.join(self.repository_dir, self.output_name, exp_datetime)
        energy_file = "energy.csv"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        tmp_result_folder = os.path.join(self.tmp_dir, exp_datetime)
        return SimpleNamespace(result_folder=result_folder, tmp_result_folder=tmp_result_folder, exp_datetime=exp_datetime, energy_file=energy_file)
    
    def _get_hyperparams(self) -> Dict:
        run_dir = self._run_dir_setup()
        hyperparams = Box({})
        hyperparams.update(**run_dir.__dict__)
        hyperparams.server = self.server.address
        for cid in range(len(self.clients)):
            hyperparams[f"client{cid}"] = self.clients[cid].address
        hyperparams.sleep_duration = 30
        hyperparams.timestamps = {}
        
        # READ YAML CONFIG FILE
        path_to_config_file = os.path.join(self.repository_dir, "config", "config_file.yaml")
        with open(path_to_config_file) as f:
            init_config = yaml.load(f, Loader=yaml.FullLoader)
        hyperparams.update(init_config)
        ip_address_full = get_host_ip(self.server.address)
        ipv4 = ip_address_full.split(" ")[0]
        logger.info("SERVER IP ADDRESS : {}".format(ipv4))
        hyperparams.comm.host = ipv4
        return hyperparams
    

        
    def _cmd_args(self, params)->str:
        """
        Convert a dictionary of parameters to a string of command line arguments
        """
        params_set = " ".join(f"{k}={v}" for k, v in params.items())
        return params_set
    
    def _cmd_host_logs(self, hparams:Box) -> None:
        nodes = self.nodes #self._get_kwargs("nodes")
        cmd = f"mkdir -p {hparams.tmp_result_folder}; echo -n > {hparams.tmp_result_folder}/logs.log;"
        _ = execute_command_on_server_and_clients(nodes, cmd, background=False)
        
    def _cmd_host_energy(self,hparams:Box) -> List[SshProcess]:
        """
        Start monitoring processes
        Args:
            hparams: hyperparameters
        Returns:
            jtop_processes: list of jtop processes
        """
        nodes = self.nodes #self._get_kwargs("nodes")
        cmd = f"python3 {self.jetson_sensor_monitor_file} --log-dir {hparams.tmp_result_folder} >> {hparams.tmp_result_folder}/logs.log 2>&1" #TO VERIFY IF LOG CORRECTLY CREATED
        jtop_processes = execute_command_on_server_and_clients(nodes, cmd, background=True)
        hparams.timestamps.start_experiment_before_sleep = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        time.sleep(hparams.sleep_duration)
        hparams.timestamps.start_experiment = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return jtop_processes
        
    def _params_to_dict(self, params:Dict)->Dict:
        """
        Convert a dictionary of parameters define as a.b.c to a nested dictionary
        Example: {"a.b.c": 1} -> {"a": {"b": {"c": 1}}}
        """
        dict_params = {}
        for k,v in params.items():
            tmp_key = k.split(".")
            subdict = dict_params
            for subkey in tmp_key[:-1]:
                if subkey not in subdict:
                    subdict[subkey] = {}
                subdict = subdict[subkey]
            subdict[tmp_key[-1]] = v
        #print("CHECK CID {}".format(dict_params["client"]))
        return dict_params
    
    def _default_params(self, params:Dict)->Dict:
        """ 
        Replace a list of dictionaries by a single dictionary
        Example: {"a.b.c": [{"d": 1}, {"e": 2}]} -> {"a.b.c": {"d": 1, "e": 2}}
        """
        new_params = {}
        for k,v in params.items():
            if isinstance(v, list):
                remplace_dict = {}
                for element in v:
                    if isinstance(element, dict):
                        remplace_dict.update(element)
                new_params.update(remplace_dict)
                logger.info("KEY {} REPLACED".format(k))
            else:
                new_params[k] = v
        return new_params
                
    
    def _hyperparams_to_csv(self, hparams:Box):
        """
        Save hyperparameters to experiment summary csv file
        """
        hparams_dict = self._default_params(hparams.to_dict())
        hparams_norm = pd.json_normalize(hparams_dict, sep='.')
        key_to_remove = self.extra_args.key_to_remove
        for k in key_to_remove:
            hparams_norm = hparams_norm.drop(columns=k)
        if os.path.exists(self.exp_csv_file):
            df = pd.read_csv(self.exp_csv_file)
            df = pd.concat([df, hparams_norm], axis=0, ignore_index=True)
        else:
            df = pd.DataFrame(hparams_norm)
        
        df.to_csv(self.exp_csv_file, index=False)
        return hparams_dict
        
    def kill_all(self)->None:
        """
        Kill server and client processes when ended
        """
        if self.run_server.ended:
            self.run_server.kill()
        for run_client in self.run_clients:
            if run_client.ended:
                run_client.kill()
        logger.info("ALL PROCESSES KILLED")
    
    def frontend_dry_run(self):
        sweeper = sweep(self.params)
        sweep_dir = os.path.join(self.repository_dir,self.sweep_name)
        self.sweeper = ParamSweeper(persistence_dir=sweep_dir, sweeps=sweeper)
        exp_count = 0
        while len(self.sweeper.get_remaining()) > 0:
            params = self.sweeper.get_next()
            logger.info("Experiment {} Remaining : {}".format(exp_count, len(self.sweeper.get_remaining())))
            logger.info("GETTING PARAMETERS")
            self.experiment_summary()
            #curr_run_args = self._run_dir_setup()
            cmd_args = self._cmd_args(params) 
            hparams = self._get_hyperparams()
            #hparams.exp_datetime = curr_run_args.exp_datetime
            #jtop_processes = self._cmd_host_energy(hparams)
            hparams.merge_update(**self._params_to_dict(params))
            logger.info("HYPERPARAMS DONE")
            hparams.timestamps.end_experiment = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            time.sleep(hparams.sleep_duration)
            hparams.timestamps.end_experiment_after_sleep = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # SAVE HYPERPARAMS
            logger.info("SAVING HYPERPARAMS TO CSV")
            debug_hp = self._hyperparams_to_csv(hparams)
        shutil.rmtree(sweep_dir) 
    
            
    def run(self):
        sweeper = sweep(self.params)
        sweep_dir = os.path.join(self.repository_dir,self.sweep_name)
        self.sweeper = ParamSweeper(persistence_dir=sweep_dir, sweeps=sweeper) 
        logger.info("Experiments left : {}".format(len(self.sweeper.get_remaining())))
        exp_count = 0
        self.experiment_summary()
        while len(self.sweeper.get_remaining()) > 0:
            print('*'*100)
            params = self.sweeper.get_next()
            logger.info("Experiment {} Remaining : {}".format(exp_count, len(self.sweeper.get_remaining())))
            
            # DEFINE HYPERPARAMS
            logger.info("GETTING PARAMETERS")
            cmd_args = self._cmd_args(params) 
            hparams = self._get_hyperparams()
            
            self._cmd_host_logs(hparams)

            logger.info("SSH PYTHON CMD TO SERVER AND CLIENTS")
            # DEFINE SERVER & CLIENT SSH PROCESSES
            server_cmd = f"cd {self.repository_dir};"\
                f"python3 main_server.py {cmd_args} comm.host={hparams.comm.host} hydra.run.dir={hparams.tmp_result_folder} >> {hparams.tmp_result_folder}/logs.log 2>&1"
            self.run_server = SshProcess(server_cmd, host=self.server, connection_params={'user': 'root'})    
            self.run_clients = []
            for (host, cid) in zip(self.clients, range(len(self.clients))):
                client_cmd = f"cd {self.repository_dir};"\
                    f"python3 client.py {cmd_args} comm.host={hparams.comm.host} hydra.run.dir={hparams.tmp_result_folder} client.cid={cid} >> {hparams.tmp_result_folder}/logs.log 2>&1"
                run_client = SshProcess(client_cmd, host=host, connection_params={'user': 'root'})
                self.run_clients.append(run_client)
            
            logger.info("START MONITORING")
            jtop_processes = self._cmd_host_energy(hparams)
            
            # RUN SERVER & CLIENTS
            logger.info("START SERVER AND CLIENTS")
            self.run_server.start()
            time.sleep(5)
            for run_client in self.run_clients:
                run_client.start()
            
            # WAIT UNTIL TRAINING IS FINISHED
            logger.info("WAITING FOR SERVER AND CLIENTS TO FINISH TRAINING")
            self.run_server.wait()
            # KILL MONITORING
            time.sleep(10) 
            for proc in jtop_processes:
                proc.kill()
            logger.info("JTOP PROCESSES FINISHED AND KILLED")
            
            # UPDATE PARAMS IN HYPERPARAMS
            logger.info("SAVE TMP FILES AND PARAMETERS TO FRONTEND")
            hparams.merge_update(**self._params_to_dict(params))
            
            server_cp_cmd = f"mkdir -p {hparams.result_folder}/server; cp -r {hparams.tmp_result_folder}/. {hparams.result_folder}/server"
            execute_command_on_server_and_clients([self.server], server_cp_cmd, background=False)
            for cid in range(len(self.clients)):
                client_cp_cmd = f"mkdir -p {hparams.result_folder}/client_{cid}; cp -r {hparams.tmp_result_folder}/. {hparams.result_folder}/client_{cid}"
                execute_command_on_server_and_clients([self.clients[cid]], client_cp_cmd, background=False)
            
            hparams.timestamps.end_experiment = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            time.sleep(hparams.sleep_duration)
            hparams.timestamps.end_experiment_after_sleep = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logger.info("FINISHED SAVING TO FRONTEND")
            
            # SAVE HYPERPARAMS
            logger.info("SAVE HYPERPARAMS TO CSV")

            debug_hp = self._hyperparams_to_csv(hparams)

            logger.info("EXPERIMENT {} DONE".format(exp_count))
            
            #params["exp_count"] = exp_count    
            exp_count += 1
        
            self.kill_all()
            
        logger.info("ALL EXPERIMENTS DONE")
        shutil.rmtree(sweep_dir)
        #return debug_hp
        

nodes = get_oar_job_nodes(448524, "toulouse")

test_params = {
    "client.lr" : [1e-1,1e-2,1e-3],
    "client.local_epochs": [1],
    "client.decay_rate": [0.1],
    "client.decay_steps": [10],
    
    "neuralnet":["MobileNetV3Small"],
    "strategy": ["fedavg"],
    "optimizer": ["SGD"],
}

to_remove = ["client.cid","client.dry_run","params.root_data","tmp_result_folder","energy_file","exp_datetime"]

Exps = MyExperiment(params=test_params,nodes=nodes,key_to_remove=to_remove)
Exps.run()
