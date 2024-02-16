from functools import reduce
import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Callable, Union, Dict
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)

def aggregate(results : List[Tuple[NDArrays, int]]) -> NDArrays:
    total_examples = sum([num_examples for _, num_examples in results])
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in results
    ]
    weights_prime = [
        reduce(np.add, layer_update)/total_examples for layer_update in zip(*weighted_weights)
    ]
    return weights_prime
    

class FedRecon(FedAvg):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
    def num_fit_clients(self, num_available_clients: int) -> int:
        num_clients = int(self.fraction_fit * num_available_clients)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluate_clients(self, num_available_clients: int) -> int:
        num_clients = int(self.fraction_evaluate * num_available_clients)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        #self.initial_parameters = None  # Don't keep initial parameters in memory
        self.g_t = self.initial_parameters
        return initial_parameters
    
    def __repr__(self) -> str:
        return "FedRecon"
    
    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(sample_size, min_num_clients)
        return [(client, fit_ins) for client in clients]
        
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy,FitRes],BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for (_, fit_res) in results
        ]
        aggregated_weights = aggregate(weights_results)
        self.g_t -= 0.1 * aggregated_weights
        parameters_aggregated = ndarrays_to_parameters(self.g_t)
        
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _ ,res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        return parameters_aggregated, metrics_aggregated
        
    # def configure_evaluate(self, config):
    #     pass
    
    # def aggregate_evaluate(self, config):
    #     pass
    
    # def evaluate(self, config):
    #     pass