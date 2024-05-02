from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from model import SimpleNet, train, train_constraints, test
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models import efficientnet, AlexNet
from optimizers.ConstraintsOpt import SFW
from optimizers.ConstraintsSet import create_lp_constraints


class SimpleClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.constraint_lp = create_lp_constraints(self.model, ord=1)     
        
    def set_parameters(self, parameters:NDArrays)->None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
        self.model.load_state_dict(state_dict, strict = True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for name,val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        """Train the model on the locally held training set."""
        self.set_parameters(parameters)
        local_epochs = int(config["local_epochs"])
        lr = float(config["lr"])
        #optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        optim = SFW(self.model.parameters(), lr=lr, momentum=0.9)
        result = train_constraints(self.model, self.trainloader, local_epochs, optim, self.constraint_lp, self.device)
        num_examples = len(self.trainloader.dataset)
        parameters_prime = self.get_parameters(config)
        return parameters_prime, num_examples, result
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, verbose=False)
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}
    
def generate_client(model, trainloader, valloader):
    def client_fn(client_id):
        return SimpleClient(model, trainloader[int(client_id)], valloader[int(client_id)]).to_client()
    return client_fn


    
    