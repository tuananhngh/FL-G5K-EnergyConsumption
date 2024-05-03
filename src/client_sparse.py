import datetime
import csv
import torch
import flwr as fl
import hydra
import logging
import os
import numpy as np

from utils.training import train_constraints, test, seed_everything
from utils.datahandler import load_clientdata_from_file
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

from optimizers.ConstraintsOpt import SFW
from optimizers.ConstraintsSet import make_feasible
from utils.serialization import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

seed_val = 2024
seed_everything(seed_val)

class ClientSparse(fl.client.Client):
    def __init__(self, model, trainloader, valloader, device, outputdir, cid, optimizer, constraints, save_model) -> None:
        self.trainloader = trainloader
        self.model = model
        self.valloader = valloader
        self.device = device
        self.outputdir = outputdir
        self.cid=cid
        self.optim = optimizer
        self.constraints = constraints
        self.save_model = save_model #feature not implemented for client yet
        
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:

        # Get parameters as a list of NumPy ndarray's
        
        ndarrays: List[np.ndarray] = self.get_parameters_internal()
    
        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_sparse_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )
    
    def get_parameters_internal(self)-> List[np.ndarray]:
        return [val.cpu().numpy() for name, val in self.model.state_dict().items()]
        
    def set_parameters(self, parameters:NDArrays)->None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
        self.model.load_state_dict(state_dict, strict = True)
        
    def write_time_csv(self,path, client_id, round, call, status):
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell()==0:
                writer.writerow(["Client ID", "Server Round","Time","Call","Status"])
            now = datetime.datetime.now()
            writer.writerow([client_id, round, now.strftime("%Y-%m-%d %H:%M:%S.%f"), call, status])
            
    def fit(self, ins:FitIns)->FitRes:
        path_comm = Path(self.outputdir, f'client_{self.cid}_comm.csv')
        self.write_time_csv(path_comm, self.cid, ins.config["server_round"], "fit", "start")
        
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)
        self.set_parameters(ndarrays_original)
        
        local_epochs:Scalar = ins.config["local_epochs"] #config["local_epochs"]
        lr:Scalar = ins.config['lr']     
        server_round = ins.config["server_round"]
        optim = instantiate(self.optim, self.model.parameters(), lr=lr)

        path = Path(self.outputdir, f'fittimes_client_{self.cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell()==0:
                writer.writerow(["Client ID", "Server Round", "Start Time", "End Time","LR", "Local Epochs"])
            start_time = datetime.datetime.now()
            log(INFO, "CLIENT {} FIT ROUND {}".format(self.cid,server_round))
            result = train_constraints(self.model, 
                                       self.trainloader, 
                                       self.valloader,
                                       local_epochs, 
                                       optim, 
                                       self.constraints, 
                                       self.device)
            end_time = datetime.datetime.now()
            log(INFO, "CLIENT {} END FIT ROUND {}".format(self.cid,server_round))
            writer.writerow([self.cid, server_round, start_time.strftime("%Y-%m-%d %H:%M:%S.%f"), end_time.strftime("%Y-%m-%d %H:%M:%S.%f"), lr, local_epochs])
        
        path = Path(self.outputdir, f'fitresult_client_{self.cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["time", "server_round", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "local_epochs"])
            now = datetime.datetime.now()
            writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S.%f"), server_round, result["train_loss"], result["train_acc"], result["val_loss"], result["val_acc"], lr, local_epochs])
            
        num_samples = len(self.trainloader.dataset)
        ndarrays_updated = self.get_parameters_internal()
        # Serialize ndarray's into a Parameters object using our custom function
        parameters_updated = ndarrays_to_sparse_parameters(ndarrays_updated)

        # Build and return response
        self.write_time_csv(path_comm, self.cid, ins.config["server_round"], "fit", "end")
        status = Status(code=Code.OK, message="Success")
        self.write_time_csv(path_comm, self.cid, server_round, "fit", "end")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=num_samples,
            metrics=result,
        )
    
    def save_sparsity(self, params:NDArrays, server_round:int):
        sparsity_dict = {}
        for i,weight in enumerate(params):
            if len(weight.shape) > 1:
                sparsity = np.count_nonzero(weight)/weight.size
                sparsity_dict[f'layer_{i}'] = sparsity
        with open(os.path.join(self.outputdir,'sparsity.log'), 'a') as f:
            f.write(f"Sparsity {server_round} : "+str(sparsity_dict) + "\n")
        
    def evaluate(self, ins:EvaluateIns) -> EvaluateRes:
        path_comm = Path(self.outputdir, f'client_{self.cid}_comm.csv')
        self.write_time_csv(path_comm, self.cid, ins.config["server_round"], "evaluate", "start")
        steps = None
        server_round = ins.config["server_round"]
        # Deserialize Parameters 
        parameters_original = ins.parameters      
        nd_arrays = sparse_parameters_to_ndarrays(parameters_original)
        #save sparsity
        self.save_sparsity(nd_arrays, server_round)
        self.set_parameters(nd_arrays)
        loss, accuracy = test(self.model, self.valloader, self.device, steps=steps, verbose=True)
        
        path = Path(self.outputdir, f'evalresult_client_{self.cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["time", "server_round", "loss", "accuracy"])
            now = datetime.datetime.now()
            writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S.%f"), server_round, loss, accuracy]) 
        num_examples = len(self.valloader.dataset)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        self.write_time_csv(path_comm, self.cid, server_round, "evaluate", "end")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=num_examples,
            metrics={"accuracy": accuracy},
        )
        
    def client_dry_run(self, config, device):
        local_epochs= config["local_epochs"]
        lr = config["lr"]
        optim = instantiate(self.optim, self.model.parameters(), lr=lr)
        logging.info(len(self.trainloader.dataset))
        result = train_constraints(self.model, 
                                   self.trainloader, 
                                   self.valloader, 
                                   local_epochs, 
                                   optim,
                                   self.device)
        return result
    
def save_client_pid(pid, client_id):
    path = Path(HydraConfig.get().runtime.output_dir, 'client_pids.csv')
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["client_id", "pid"])
        writer.writerow([client_id, pid])
        
@hydra.main(config_path="config", config_name="config_file",version_base=None)
def main(cfg:DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client_id = cfg.client.cid
    dry_run = cfg.client.dry_run
    num_classes = cfg.params.num_classes
    save_model = cfg.params.save_model
    host = cfg.comm.host
    port = cfg.comm.port
    output_dir = HydraConfig.get().runtime.output_dir
    server_address = str(host)+":"+str(port)

    # get pid of the current process
    pid = os.getpid()
    # save client pid
    save_client_pid(pid, client_id)
    
    trainloader, valloader = load_clientdata_from_file(cfg.data, client_id)
    print(len(trainloader.dataset))

    model = instantiate(cfg.neuralnet)
    model = convert_bn_to_gn(model, num_groups=cfg.params.num_groups)
    #constraints
    constraints = instantiate(cfg.constraints, model=model)
    make_feasible(model, constraints)
    
    optimizer = cfg.optimizer
    client = ClientSparse(model, 
                          trainloader, 
                          valloader, 
                          device, 
                          output_dir, 
                          client_id, 
                          optimizer, 
                          constraints, 
                          save_model)
    if dry_run:
        res = client.client_dry_run(cfg.client_params, device)
        logging.info(res)
    else:
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