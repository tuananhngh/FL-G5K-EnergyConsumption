from logging import INFO, DEBUG
import argparse
from typing import List, Tuple
import torch
import flwr as fl
from collections import OrderedDict
from typing import Dict
from flwr.common.logger import log
from flwr.client import Client
from matrix_factorization import MatrixFactorizationModel, build_reconstruction_model
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


def get_parameters(model)->Tuple[List[NDArrays], List[NDArrays]]:
    global_params = []
    local_params = []
    log(INFO, "FUNCTION GET_PARAMETERS CALLED")
    for name,val in model.state_dict().items():
        if "global_layers" in name:
            global_params.append(val.cpu().numpy())
        elif "local_layers" in name:
            local_params.append(val.cpu().numpy())
    return global_params, local_params

def set_parameters(model, global_params:List[NDArrays])->None:
    log(INFO, "FUNCTION SET_PARAMETERS CALLED")
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
        log(INFO, "Get Parameters called in as CLIENT METHOD")
        ndarray_global, _ = get_parameters(self.model)
        #ndarray_global = parameters
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
        log(INFO, f"Client {self.client_cid} Fit, recon_loss: {recon_loss}, pers_loss: {pers_loss}")
        updated_global_params, _  = get_parameters(pers_model)
        log(INFO, "Get Parameters called in Fit method")
        
        global_array_diff_updated = [updated - original for updated,original in zip(updated_global_params, original_params)]
        
        parameters_updated = ndarrays_to_parameters(global_array_diff_updated)
        
        status = Status(code=Code.OK, message="Success")
        
        return FitRes(status=status,
                      parameters=parameters_updated,
                      num_examples=len(self.trainloader.dataset),
                      metrics={})
        
    def evaluate(self, ins:EvaluateIns)->EvaluateRes:
        original_params = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.model, original_params)
        loss = test(self.model, self.valloader)
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples = len(self.valloader.dataset),
            metrics = {"loss":loss}
        )
        
    def client_dry_run(self, model, client_cid, trainloaders, valloaders,device):
        log(INFO, f"Client {client_cid} dry run")
        recon_epoch = 3
        pers_epoch = 3
        recon_model, recon_loss = recon_train(model, recon_epoch, trainloaders, 0.01)
        pers_model, pers_loss = pers_train(recon_model, pers_epoch, trainloaders, 0.01)
        loss = test(pers_model, valloaders)
        log(INFO, f"Client {client_cid} dry run loss: {loss}")
        return recon_loss, pers_loss      
        


def recon_train(model, recon_epochs, support_dataloader, lr):
    criterion = torch.nn.MSELoss()
    optimizer_recon = torch.optim.SGD(model.local_layers.parameters(), lr=lr,
                                      weight_decay=0.)
    user = torch.tensor([0], dtype=torch.long).to(model.device) #for compatible with model
    for epoch in range(recon_epochs):
        running_loss = 0.0
        for inputs, targets in support_dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            item = inputs
            #print("Item : {}".format(item.shape))
            outputs = model(user, item)
            #log(INFO, f"Outputs : {outputs}")
            loss = criterion(outputs, targets)
            #log(INFO, f"Loss : {loss}"
            loss.backward()
            optimizer_recon.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(support_dataloader)
        log(INFO, f"Epoch {epoch+1} Recon loss: {epoch_loss}")
    return model, epoch_loss

def pers_train(model, pers_epochs, query_dataloader, lr):
    criterion = torch.nn.MSELoss()
    optimizer_pers = torch.optim.SGD(model.global_layers.parameters(), lr=lr,
                                     weight_decay=0.)
    user_input = torch.tensor([0], dtype=torch.long).to(model.device) #for compatible with model
    for epoch in range(pers_epochs):
        running_loss = 0.
        for inputs, targets in query_dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            item_input = inputs
            outputs = model(user_input, item_input)
            
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
        user = torch.tensor([0], dtype=torch.long).to(model.device)
        for inputs, targets in testloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            item = inputs
            outputs = model(user, item)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
        log(INFO, f"Val loss: {running_loss/len(testloader)}")
    return running_loss/len(testloader)

from load_movielens import create_user_dataloader, load_movielens_data, create_user_datasets, split_dataset, path_to_1m

def main(client_cid, dry_run=False):
    ratings_df, movies_df = load_movielens_data(path_to_1m)
    num_users, num_items = len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
    user_datasets = create_user_datasets(ratings_df, min_examples_per_user=50, max_clients=4000)
    #train_users,val_users, test_users = split_dataset(user_datasets, 0.8, 0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    trainloader, valloader, testloader = create_user_dataloader(user_datasets[client_cid], 0.8, 0.1, 5)
    model, global_params, local_params = build_reconstruction_model(num_users=1, num_items=num_items, num_latent_factors=50, personal_model=True, add_biases=False, l2_regularizer=0.0, spreadout_lambda=0.0)
    
    client = Client_ML(model, trainloader, valloader, device, client_cid)
    if dry_run:
        client.client_dry_run(model, client_cid, trainloader, valloader, device)
    else:
        fl.client.start_client(
            server_address="[::]:8080",
            client = client
        )
    #client.client_dry_run(model, client_cid, trainloader, valloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",type=int, help="Client ID")
    args = parser.parse_args()
    
    client_cid = args.id
    dry_run = False
    main(client_cid, dry_run)