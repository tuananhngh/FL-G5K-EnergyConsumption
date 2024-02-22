from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from model import SimpleNet, train, test, Net
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models import efficientnet, AlexNet



class SimpleClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = mobilenet_v3_small(num_classes=10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # def set_parameters(self, parameters: NDArrays) -> None:
    #     """Set model parameters from a list of NumPy ndarrays."""
    #     params_dict = zip(self.model.state_dict().keys(), parameters)
    #     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    #     self.model.load_state_dict(state_dict, strict=True)
        
    def set_parameters(self, parameters:NDArrays)->None:
        key = [k for k in self.model.state_dict().keys() if 'num_batches_tracked' not in k]
        #params_dict = zip(self.model.state_dict().keys(), parameters)
        params_dict = zip(key, parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
        #state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})
        self.model.load_state_dict(state_dict, strict = False)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model paramaters for server"""
        #return [val.cpu().numpy() for name,val in self.model.state_dict().items() if 'bn' not in name]
        return [val.cpu().numpy() for name,val in self.model.state_dict().items() if 'num_batches_tracked' not in name]
    def fit(self, parameters, config):
        """Train the model on the locally held training set."""
        self.set_parameters(parameters)
        local_epochs :int = config["local_epochs"]
        lr = config["lr"]
        optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        result = train(self.model, self.trainloader, local_epochs, optim, self.device)
        num_batches = len(self.trainloader)
        parameters_prime = self.get_parameters(config)
        return parameters_prime, num_batches, {"train_acc": result["train_acc"], "train_loss": result["train_loss"]}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, verbose=False)
        return float(loss), len(self.valloader), {"accuracy": accuracy}
    
def generate_client(trainloader, valloader):
    def client_fn(client_id):
        return SimpleClient(trainloader[int(client_id)], valloader[int(client_id)]).to_client()
    return client_fn


    
    