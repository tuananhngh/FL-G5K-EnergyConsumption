import flwr as fl
from typing import List, Tuple, Optional, Callable, Union, Dict
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.common import (
    Strategy,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class FedRecon(Strategy):
    def __init__(self):
        pass
    
    def __repr__(self) -> str:
        return "FedRecon"
    
    def num_fit_clients(self, num_available_clients: int) -> int:
        num_clients = int(self.fraction_fit * num_available_clients)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluate_clients(self, num_available_clients: int) -> int:
        num_clients = int(self.fraction_evaluate * num_available_clients)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    
    def initialize_parameters(self) -> Parameters:
        initial_params = self.initial_parameters
        self.initialize_parameters = None
        return initial_params 
    
    def __repr__(self) -> str:
        return "FedRecon"
    
    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        pass
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy,FitRes],BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        global_ = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for (_, fit_res) in results
        ]
        
        
    def configure_evaluate(self, config):
        pass
    
    def aggregate_evaluate(self, config):
        pass
    
    def evaluate(self, config):
        pass