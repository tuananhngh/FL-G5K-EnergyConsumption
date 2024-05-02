from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar
import numpy as np
import torch
import flwr as fl
from model import SimpleNet, train, train_constraints, test
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models import efficientnet, AlexNet
from optimizers.ConstraintsOpt import SFW
from optimizers.ConstraintsSet import create_lp_constraints
from serialization import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays
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


class Client(fl.client.Client):
    def __init__(self, model, constraints, trainloader, valloader) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.constraint_lp = constraints
        
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
        
    def fit(self, ins) -> FitRes:
        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_original = ins.parameters
        ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)
        
        # Update local model, train, get updated parameters
        self.set_parameters(ndarrays_original)
        local_epochs:Scalar = ins.config["local_epochs"] #config["local_epochs"]
        lr:Scalar = ins.config['lr']     
        #optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        optim = SFW(self.model.parameters(), lr=lr, momentum=0.9) #type: ignore
        result = train_constraints(self.model, self.trainloader, local_epochs, optim, self.constraint_lp, self.device)
        num_examples = len(self.trainloader.dataset)
        ndarrays_updated = self.get_parameters_internal()

        # Serialize ndarray's into a Parameters object using our custom function
        parameters_updated = ndarrays_to_sparse_parameters(ndarrays_updated)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=num_examples,
            metrics=result,
        )

    def evaluate(self, ins:EvaluateIns) -> EvaluateRes:
        # Deserialize Parameters 
        parameters_original = ins.parameters
        nd_arrays = sparse_parameters_to_ndarrays(parameters_original)
        self.set_parameters(nd_arrays)
        loss, accuracy = test(self.model, self.valloader, verbose=False)
        num_examples = len(self.valloader.dataset)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=num_examples,
            metrics={"accuracy": accuracy},
        )
        
def generate_client(model, constraints, trainloader, valloader):
    def client_fn(client_id):
        return Client(model, constraints, trainloader[int(client_id)], valloader[int(client_id)])
    return client_fn
        