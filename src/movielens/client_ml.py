from logging import INFO, DEBUG
from typing import List, Tuple
import torch
import flwr as fl
from collections import OrderedDict
from typing import Dict
from flwr.common.logger import log
from flwr.client import Client
from flwr.common import (
    NDArrays,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Status,
    EvaluateIns,
    EvaluateRes,
    Code
)

#logger = logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_parameters(model)->Tuple[List[NDArrays], List[NDArrays]]:
    global_params = []
    local_params = []
    for name,val in model.state_dict().items():
        if "global_layers" in name:
            global_params.append(val.cpu().numpy())
        elif "local_layers" in name:
            local_params.append(val.cpy().numpy())
    return global_params, local_params

def set_parameters(model, global_params:List[NDArrays])->None:
    keys = [k for k in model.state_dict().keys() if 'global_layers' in k]
    global_state_dict = OrderedDict({
            k:torch.tensor(v) for k,v in zip(keys, global_params)
        })
    model.load_state_dict(global_state_dict, strict=False)
    
class Client_ML(fl.client.Client):
    def __init__(self, model, trainloader, valloader, device, client_cid):
        self.model = model
        self.device = device
        self.client_cid = client_cid
        self.trainloader = trainloader
        self.valloader = valloader
        
    def get_parameters(self, ins:GetParametersIns) -> GetParametersRes:
        """
        Get the current model global parameters difference and send back to the server
        """
        ndarray_global, _ = get_parameters(self.model)
        parameters = ndarrays_to_parameters(ndarray_global)
        
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)
        
        
    def set_parameters(self, ins:FitIns)->None:
        """
        Receive global parameters from server

        Args:
            ins (FitsIns): _description_
        """
        params_server = ins.parameters
        ndarray_server = parameters_to_ndarrays(params_server)
        set_parameters(self.model, ndarray_server)
        
    def fit(self, ins:FitIns)->FitRes:
        log(INFO, f'Client {self.client_cid} Fit, config: {ins.config}')
        recon_epochs = ins.config['recon_epochs']
        pers_epochs = ins.config['pers_epochs']
        recon_lr, pers_lr = ins.config['recon_lr'], ins.config['pers_lr']
        original_params = parameters_to_ndarrays(ins.parameters) #Original global parameters, convert to ndarrays
        set_parameters(self.model, original_params) #Set client global layers
        
        recon_model, recon_loss = recon_train(self.model, recon_epochs, self.trainloader, recon_lr) #Train local
        pers_model, pers_loss = pers_train(recon_model, pers_epochs, self.trainloader, pers_lr) #Train global
        #Make difference between trained global and global original
        updated_global_params, _  = get_parameters(pers_model)
        
        global_array_diff_updated = [updated - original for updated,original in zip(updated_global_params, original_params)]
        
        parameters_updated = ndarrays_to_parameters(global_array_diff_updated)
        
        status = Status(code=Code.OK, message="Success")
        
        return FitRes(status=status,
                      parameters=parameters_updated,
                      num_examples=len(self.trainloader),
                      metrics={})
        
    def evaluate(self, ins:EvaluateIns)->EvaluateRes:
        original_params = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.model, original_params)
        loss, accuracy = test(self.model, self.valloader)
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples = len(self.valloader),
            metrics = {"accuracy":accuracy}
        )
        
    def client_dry_run(self, model, client_id, trainloaders, valloaders, config, device):
        recon_epoch = 1
        pers_epoch = 1
        
        

def recon_train(model, recon_epochs, support_dataloader, lr):
    criterion = torch.nn.MSELoss()
    optimizer_recon = torch.optim.SGD(model.local_layers.parameters(), lr=lr)
    for epoch in range(recon_epochs):
        running_loss = 0.0
        for inputs, targets in support_dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            user,item = inputs
            outputs = model(user, item)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_recon.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(support_dataloader)
        log(INFO, f"Epoch {epoch+1} Recon loss: {epoch_loss}")
    return model, epoch_loss

def pers_train(model, pers_epochs, query_dataloader, lr):
    criterion = torch.nn.MSELoss()
    optimizer_pers = torch.optim.SGD(model.global_layers.parameters(), lr=lr)
    for epoch in range(pers_epochs):
        running_loss = 0.
        for inputs, targets in query_dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            user, item = inputs
            outputs = model(user, item)
            
            loss = criterion(outputs, targets)
            optimizer_pers.zero_grad()
            loss.backward()
            optimizer_pers.step()
            
            running_loss += loss.item()
        epoch_loss = running_loss/len(query_dataloader)
        log(INFO, f"Epoch {epoch+1} Pers loss: {epoch_loss}")
    return model, epoch_loss


def test(model, testloader):
    with torch.no_grad():
        criterion = torch.nn.MSELoss()
        running_loss = 0.
        for inputs, targets in testloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            user, item = inputs
            outputs = model(user, item)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
        return running_loss/len(testloader)