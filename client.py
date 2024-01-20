import datetime
import os
import torch
import torch.nn as nn
import flwr as fl
import utils
import hydra
import models
from pathlib import Path
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, Tuple, List
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, HydraConfig
from logging import INFO, DEBUG
from flwr.common.logger import log
import jetson_monitoring_energy



class Client(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device, outputdir, cid) -> None:
        self.trainloader = trainloader
        self.model = model
        self.valloader = valloader
        self.device = device
        self.outputdir = outputdir
        self.cid=cid
    
    def set_parameters(self, parameters:NDArrays)->None:
        key = [k for k in self.model.state_dict().keys()]
        params_dict = zip(key, parameters)
        #state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
        state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})
        self.model.load_state_dict(state_dict, strict = True)
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for name,val in self.model.state_dict().items()]
    
    def fit(self, parameters, config:Dict[str,Scalar]=None) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on the locally held training set."""
        self.set_parameters(parameters)
        local_epochs = config["local_epochs"]
        lr = config["lr"]
        server_round= config["server_round"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        log(INFO, "CLIENT {} FIT ROUND {}".format(self.cid,server_round))
        result = utils.train(self.model, self.trainloader, self.valloader, local_epochs, optim, self.device)
        log(INFO, "CLIENT {} END FIT ROUND {}".format(self.cid,server_round))
        # Write result to file with current date and time
        path = Path(self.outputdir, 'fitresult.txt')
        with open(path, 'a') as f:
            now = datetime.datetime.now()
            f.write(f'\n{now.strftime("%Y-%m-%d %H:%M:%S")} - Result Round {server_round}: {result}')
        num_samples = len(self.trainloader.dataset)
        parameters_prime = self.get_parameters(config)
        return parameters_prime, num_samples, result
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the locally held test dataset."""
        steps = None #config["test_steps"]
        server_round = config["server_round"]
        self.set_parameters(parameters)
        loss, accuracy = utils.test(self.model, self.valloader, self.device, steps=steps,verbose=True) 
        path = Path(self.outputdir, 'evalresult.txt')
        with open(path, 'a') as f:
            now = datetime.datetime.now()
            f.write(f'\n{now.strftime("%Y-%m-%d %H:%M:%S")} - Round {server_round}, Loss : {loss}, Accuracy : {accuracy}')
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}
    
    def client_dry_run(self, model, client_id, trainloaders, valloaders, config, device):
        local_epochs= config["local_epochs"]
        lr = config["lr"]
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        print(len(self.trainloader.dataset))
        result = utils.train(model, self.trainloader, self.valloader, local_epochs, optim, self.device)
        return result

@hydra.main(config_path="config", config_name="config_file")
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client_id = cfg.client.cid
    dry_run = cfg.client.dry_run
    num_classes = cfg.params.num_classes
    host = cfg.comm.host
    port = cfg.comm.port
    output_dir = HydraConfig.get().runtime.output_dir
    server_address = str(host)+":"+str(port)
    trainloaders, valloaders, testloader = utils.load_dataset(cfg.params)
    
    #trainloader, valloader, testloader = utils.load_dataloader(client_id, path_to_data)
    model = instantiate(cfg.neuralnet)
    client = Client(model, trainloaders[client_id], valloaders[client_id], device, output_dir, client_id)
    if dry_run:
        res = client.client_dry_run(model, client_id, trainloaders, valloaders, cfg.client_params, device)
        print(res)
    else:
        fl.client.start_numpy_client(server_address=server_address, client=client)
        
if __name__ == "__main__":
    main()
    
