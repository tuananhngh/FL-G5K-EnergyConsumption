
from omegaconf import DictConfig
import torch
from collections import OrderedDict
from model import SimpleNet, test, Net
from torchvision.models.mobilenetv3 import mobilenet_v3_small

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round):
        return {'lr': config.lr, 'local_epochs': config.local_epochs}
    return fit_config_fn

def set_parameters(model, parameters)->None:
    key = [k for k in model.state_dict().keys() if 'num_batches_tracked' not in k]
    #params_dict = zip(model.state_dict().keys(), parameters)
    params_dict = zip(key, parameters)
    state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
    #state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})
    model.load_state_dict(state_dict, strict = True)
    return model

def get_evaluate_config(model, testloader):
    def evaluate_fn(server_round, parameters, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        set_parameters(model, parameters)
        loss, accuracy = test(model, testloader, verbose=False) 
        return loss, {"accuracy": accuracy}
    return evaluate_fn