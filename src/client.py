import datetime
import csv
import torch
import flwr as fl
import hydra
import logging
import os
import gc
from utils.training import train, test, seed_everything, train_constraints
#from utils.datahandler import load_clientdata_from_file
from utils.datafunctions import load_clientdata_from_file
from utils.models import convert_bn_to_gn
from pathlib import Path
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, Tuple, List
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, HydraConfig
from logging import INFO, DEBUG
from flwr.common.logger import log
import sys
sys.path.append("/home/tunguyen/jetson-imagenet/src/utils")

seed_val = 2024
seed_everything(seed_val)

class Client(fl.client.NumPyClient):
    def __init__(self, model, cfg, device, outputdir, optimizer, save_model, constraints) -> None:
        self.main_cfg = cfg
        self.model = model
        self.device = device
        self.outputdir = outputdir
        self.optim = optimizer
        self.save_model = save_model #feature not implemented for client yet
        self.constraints = constraints
    
    def set_parameters(self, parameters:NDArrays)->None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
        self.model.load_state_dict(state_dict, strict = True)
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for name,val in self.model.state_dict().items()]
    
    def write_time_csv(self,path, client_id, round, call, status):
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell()==0:
                writer.writerow(["Client ID", "Server Round","Time","Call","Status"])
            now = datetime.datetime.now()
            writer.writerow([client_id, round, now.strftime("%Y-%m-%d %H:%M:%S.%f"), call, status])
            
    def get_dataloader_cid(self, client_id):
        # get pid of the current process
        pid = os.getpid()
        # save client pid
        save_client_pid(pid, client_id)
        trainloader, valloader = load_clientdata_from_file(self.main_cfg.data, client_id)
        return trainloader, valloader
        
    #Todo : Reinitialize the host with different client id each communication round using : 
    # _ sampling client from strategy with a mapping from clientproxy cid to user define cid
    # _ do the same for evaluation
    # _ reintialize the host by clearing caches files
        
    
    def fit(self, parameters, config:Dict[str,Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on the locally held training set."""
        # Reinitialize the host with the new client id
        cid = config['cid'] # Get client id from server fit instruction 
        log(INFO, f"Client ID: {cid}")
        trainloader, valloader = self.get_dataloader_cid(cid)
        path_comm = Path(self.outputdir, f'client_{cid}_comm.csv')
        self.write_time_csv(path_comm, cid, config["server_round"], "fit", "start")
        self.set_parameters(parameters)

        local_epochs = config["local_epochs"]
        lr = config["lr"]
        server_round= config["server_round"]
        optim = instantiate(self.optim, self.model.parameters(), lr=lr)
        
        path = Path(self.outputdir, f'fittimes_client_{cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell()==0:
                writer.writerow(["Client ID", "Server Round", "Start Time", "End Time","LR", "Local Epochs"])
            start_time = datetime.datetime.now()
            log(INFO, "CLIENT {} FIT ROUND {}".format(cid,server_round))
            if self.constraints is not None:
                result = train_constraints(self.model, 
                                       trainloader, 
                                       valloader,
                                       local_epochs, 
                                       optim, 
                                       self.constraints, 
                                       self.device)
            else:
                result = train(self.model, 
                               trainloader, 
                               valloader, 
                               local_epochs, 
                               optim, 
                               self.device)
            end_time = datetime.datetime.now()
            log(INFO, "CLIENT {} END FIT ROUND {}".format(cid,server_round))
            writer.writerow([cid, server_round, start_time.strftime("%Y-%m-%d %H:%M:%S.%f"), end_time.strftime("%Y-%m-%d %H:%M:%S.%f"), lr, local_epochs])
        
        # Write result to file with current date and time
        path = Path(self.outputdir, f'fitresult_client_{cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["time", "server_round", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "local_epochs"])
            now = datetime.datetime.now()
            writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S.%f"), server_round, result["train_loss"], result["train_acc"], result["val_loss"], result["val_acc"], lr, local_epochs])
        num_samples = len(trainloader.dataset)
        parameters_prime = self.get_parameters(config)
        self.write_time_csv(path_comm, cid, config["server_round"], "fit", "end")
        
        # delete the dataloaders
        del trainloader
        del valloader
        gc.collect()
        # free cuda memory
        # if self.device == 'cpu':
        #     pass
        # else:
        #     with torch.cuda.device(self.device):
        #         torch.cuda.empty_cache()
        
        return parameters_prime, num_samples, result
    
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the locally held test dataset."""
        cid = config['cid']
        trainloader , valloader = self.get_dataloader_cid(cid)
        path_comm = Path(self.outputdir, f'client_{cid}_comm.csv')
        self.write_time_csv(path_comm, cid, config["server_round"], "evaluate", "start")
        steps = None #config["test_steps"]
        server_round = config["server_round"]

        self.set_parameters(parameters)
        loss, accuracy = test(self.model, valloader, self.device, steps=steps,verbose=True) 
        
        path = Path(self.outputdir, f'evalresult_client_{cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["time", "server_round", "loss", "accuracy"])
            now = datetime.datetime.now()
            writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S.%f"), server_round, loss, accuracy])
        self.write_time_csv(path_comm, cid, config["server_round"], "evaluate", "end")
        num_samples = len(valloader.dataset)
        
        # delete the dataloaders
        del trainloader
        del valloader
        gc.collect()
        # free cuda memory
        # if self.device == 'cpu':
        #     pass
        # else:
        #     with torch.cuda.device(self.device):
        #         torch.cuda.empty_cache()
        
        return float(loss), num_samples, {"accuracy": accuracy}
        

def save_client_pid(pid, client_id):
    path = Path(HydraConfig.get().runtime.output_dir, 'client_pids.csv')
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["client_id", "pid"])
        writer.writerow([client_id, pid])

@hydra.main(config_path="config", config_name="config_file",version_base=None)
def main(cfg:DictConfig):
    #logging.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_model = cfg.params.save_model
    host = cfg.comm.host
    port = cfg.comm.port
    output_dir = HydraConfig.get().runtime.output_dir
    server_address = str(host)+":"+str(port)
    
    model = instantiate(cfg.neuralnet)
    model = convert_bn_to_gn(model, num_groups=cfg.params.num_groups)
    optimizer = cfg.optimizer
    if "constraints" in cfg.keys():
        constraints = instantiate(cfg.constraints, model)
    else:
        constraints = None
    client = Client(model, cfg, device, output_dir, optimizer, save_model,constraints).to_client()
    fl.client.start_client(server_address=server_address, client=client)
        
if __name__ == "__main__":
    logging.basicConfig(
        # filename=args.log_dir + args.log_file, 
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d : %(message)s',
        )
    
    try:
        main()
    except Exception as err:
        logging.error(err)
    
