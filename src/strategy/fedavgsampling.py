from logging import WARNING, INFO
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.server.strategy.aggregate import aggregate

from utils.serialization import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays
from flwr.common import (
    parameters_to_ndarrays, 
    ndarrays_to_parameters,
    FitIns,
    FitRes,
    EvaluateIns,
)

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
import logging
import numpy as np
import os
import copy
import random


class FedAvgSampling(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        num_clients : int = 10,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.num_clients = num_clients
            
    def __repr__(self) -> str:
        return "FedAvgSampling"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        sample_size = len(client_instructions)
        num_available = client_manager.num_available()
        choose_id = random.sample(range(self.num_clients),sample_size)
        log(INFO, f"Client Fit IDs Round {server_round}: {choose_id} from {self.num_clients} clients")
        sampled_client_instructions = []
        for (cproxy, fitins),uid in zip(client_instructions,choose_id):
            client_fitins = copy.deepcopy(fitins)
            client_fitins.config['cid']=uid
            sampled_client_instructions.append((cproxy, client_fitins))
        #log(INFO, f"{sampled_client_instructions[0][0]} {sampled_client_instructions[0][1].config['cid']}, {sampled_client_instructions[1][0]} {sampled_client_instructions[1][1].config['cid']}" )
        return sampled_client_instructions
        
 
        
        
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        client_instructions = super().configure_evaluate(server_round, parameters, client_manager)
        sample_size = len(client_instructions)
        num_available = client_manager.num_available()
        choose_id = random.sample(range(self.num_clients),sample_size)
        
        log(INFO, f"Client Evaluate IDs Round {server_round}: {choose_id} from {self.num_clients} clients")
        sampled_client_instructions = []
        for (cproxy, evaluateins),uid in zip(client_instructions,choose_id):
            client_evalins = copy.deepcopy(evaluateins)
            client_evalins.config['cid']=uid
            sampled_client_instructions.append((cproxy, client_evalins))
        return sampled_client_instructions