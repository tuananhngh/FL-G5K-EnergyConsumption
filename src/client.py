import datetime
import csv
import torch
import flwr as fl
import hydra
import logging

from utils.training import train, test, seed_everything
from utils.datahandler import load_clientdata_from_file
from pathlib import Path
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict, Tuple, List
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, HydraConfig
from logging import INFO, DEBUG
from flwr.common.logger import log

seed_val = 2024
seed_everything(seed_val)

class Client(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device, outputdir, cid, optimizer) -> None:
        self.trainloader = trainloader
        self.model = model
        self.valloader = valloader
        self.device = device
        self.outputdir = outputdir
        self.cid=cid
        self.optim = optimizer
    
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
        optim = instantiate(self.optim, self.model.parameters(), lr=lr)
        #optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        
        path = Path(self.outputdir, f'fittimes_client_{self.cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell()==0:
                writer.writerow(["Client ID", "Server Round", "Start Time", "End Time","LR", "Local Epochs"])
            start_time = datetime.datetime.now()
            log(INFO, "CLIENT {} FIT ROUND {}".format(self.cid,server_round))
            result = train(self.model, self.trainloader, self.valloader, local_epochs, optim, self.device)
            end_time = datetime.datetime.now()
            log(INFO, "CLIENT {} END FIT ROUND {}".format(self.cid,server_round))
            writer.writerow([self.cid, server_round, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S"), lr, local_epochs])
            
        # Write result to file with current date and time
        path = Path(self.outputdir, f'fitresult_client_{self.cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["time", "server_round", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "local_epochs"])
            now = datetime.datetime.now()
            writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), server_round, result["train_loss"], result["train_acc"], result["val_loss"], result["val_acc"], lr, local_epochs])
        num_samples = len(self.trainloader.dataset)
        parameters_prime = self.get_parameters(config)
        return parameters_prime, num_samples, result
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the locally held test dataset."""
        steps = None #config["test_steps"]
        server_round = config["server_round"]
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device, steps=steps,verbose=True) 
        
        path = Path(self.outputdir, f'evalresult_client_{self.cid}.csv')
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["time", "server_round", "loss", "accuracy"])
            now = datetime.datetime.now()
            writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), server_round, loss, accuracy])
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}
    
    def client_dry_run(self, model, client_id, trainloaders, valloaders, config, device):
        local_epochs= config["local_epochs"]
        lr = config["lr"]
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        logging.info(len(self.trainloader.dataset))
        result = train(model, self.trainloader, self.valloader, local_epochs, optim, self.device)
        return result

@hydra.main(config_path="config", config_name="config_file")
def main(cfg:DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client_id = cfg.client.cid
    dry_run = cfg.client.dry_run
    num_classes = cfg.params.num_classes
    host = cfg.comm.host
    port = cfg.comm.port
    output_dir = HydraConfig.get().runtime.output_dir
    server_address = str(host)+":"+str(port)
    
    #dataconfig = DataSetHandler(cfg.data)
    #trainloaders, valloaders, testloader = dataconfig()
    trainloader, valloader = load_clientdata_from_file(cfg.data, client_id)
    print(len(trainloader.dataset))
    #trainloader, valloader, testloader = load_dataloader(client_id, path_to_data)
    model = instantiate(cfg.neuralnet)
    optimizer = cfg.optimizer
    client = Client(model, trainloader, valloader, device, output_dir, client_id, optimizer)
    if dry_run:
        res = client.client_dry_run(model, client_id, trainloader, valloader, cfg.client_params, device)
        logging.info(res)
    else:
        fl.client.start_numpy_client(server_address=server_address, client=client)
        
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
    
