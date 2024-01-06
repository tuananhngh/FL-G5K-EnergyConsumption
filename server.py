# Description: This file contains the code for the server. 
from collections import OrderedDict
from numpy import ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
import utils
import models
from omegaconf import DictConfig
from typing import Callable, Dict, Tuple, List
from flwr.common import Metrics, NDArray, Scalar, ndarrays_to_parameters, FitIns, EvaluateRes
from flwr.server.strategy import FedAvg, FedAdam


def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config(config: Dict[str, Scalar])->Callable:
    def fit_config_fn(server_round:int)->FitIns:
        return {'lr': config.lr, 'local_epochs': config.local_epochs, 'lr_decay': config.lr_decay}
    return fit_config_fn

def get_evaluate_fn(model, testloader, device, cfg: Dict[str, Scalar])->Callable:
    def evaluate_fn(server_round:int, parameters:NDArray, config):
        num_classes = cfg["num_classes"]
        steps = len(testloader)
        params_dict = zip(model.state_dict().keys(),parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        #utils.set_parameters(model, parameters)
        loss, accuracy = utils.test(model, testloader, device, verbose=False)
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate_fn
        
#strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# fl.server.start_server(server_address="192.168.1.110:8080",
#                        config=fl.server.ServerConfig(num_rounds=5), 
#                        strategy=FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
#                                        evaluate_fn=get_evaluate_fn(utils.load_dataloader("server", "data"), utils.get_config("config/server_config.yaml")),
#                                        initial_parameters=ndarrays_to_parameters(utils.get_parameters(utils.Net(num_classes=10))))
# )