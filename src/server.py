import csv
import datetime
from itertools import accumulate

from flwr.server.criterion import Criterion
from utils.training import get_parameters, test, seed_everything, set_parameters
from utils.datahandler import load_testdata_from_file
import hydra
import os
import numpy as np
import flwr as fl
import logging
import torch
import pickle 
import timeit
import threading
import random
from collections import OrderedDict
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from flwr.common import Metrics, NDArray, Scalar, ndarrays_to_parameters, FitIns, EvaluateRes
from hydra.utils import instantiate
from typing import Callable, Dict, Tuple, List, Optional, Union
from flwr.server.history import History
from logging import DEBUG, INFO
from flwr.common.logger import log
from utils.models import convert_bn_to_gn
from flwr.server.strategy import FedAvg, FedYogi, FedAdam
from optimizers.ConstraintsSet import make_feasible
from utils.serialization import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
import concurrent.futures
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

seed_val = 2024
seed_everything(seed_val)

def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def learning_rate_scheduler(lr, epoch, decay_rate, decay_steps):
    lr = lr * (decay_rate ** (epoch // decay_steps))
    return lr


def get_on_fit_config(config: Dict[str, Scalar])->Callable:
    def fit_config_fn(server_round:int)->FitIns:
        #logging.info(f"Server Round: {server_round}")
        decay_rate = config.decay_rate
        decay_steps = config.decay_steps
        lr = learning_rate_scheduler(config.lr, server_round, decay_rate, decay_steps)
        return {'lr': lr, 'local_epochs': config.local_epochs, 'server_round': server_round}
    return fit_config_fn
    
def get_on_evaluate_config(config: Dict[str, Scalar])->Callable:
    def evaluate_config_fn(server_round:int)->EvaluateRes:
        return {'server_round': server_round}
    return evaluate_config_fn


def get_evaluate_fn(model, testloader, device, cfg: Dict[str, Scalar], output_dir:str)->Callable:
    save_model = cfg.save_model
    def evaluate_fn(server_round:int, parameters:NDArray, config):
        model.to(device)
        set_parameters(model, parameters)
        loss, accuracy = test(model, testloader, device, verbose=False)
        if save_model:
            if server_round%50 == 0: # Save model every ... rounds
                torch.save(model.state_dict(), Path(output_dir, f"model_{server_round}.pt"))
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate_fn

def write_time_csv(path, round, call, status):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell()==0:
            writer.writerow(["round", "timestamp","call","status"])
        now = datetime.datetime.now()
        writer.writerow([round, now.strftime("%Y-%m-%d %H:%M:%S.%f"), call, status])
        
class CustomSimpleClientManager(fl.server.SimpleClientManager):
    def __init__(self)->None:
        super().__init__()
        
    # def clients_map(self) -> None:
    #     """Map clients to user defined id"""
    #     available_cids = sorted(list(self.clients))
    #     num_clients = len(available_cids)
    #     self.mapping = {cprox:uid for cprox, uid in zip(available_cids, range(num_clients))}
        
    def sample(
        self,
        num_clients: int, #client participe in training
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[Tuple[ClientProxy,int]]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        
        # Map clients to user defined id
        available_cids = list(self.clients)
        available_cids = sorted(available_cids)
        nb_available_cids = len(available_cids)
        mapping = {cid:user_id for cid, user_id in zip(available_cids, range(nb_available_cids))}
        
        # Sample clients which meet the criterion
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        sampled_cids = random.sample(available_cids, num_clients)
        user_cids = [mapping[cid] for cid in sampled_cids]
        clients_and_id = [(self.clients[cid],user_cid) for cid,user_cid in zip(sampled_cids, user_cids)]
        return clients_and_id
        
        

class CustomServer(fl.server.Server):
    def __init__(self,wait_round:int, path_log:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_round = wait_round
        self.path_log = path_log
        
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        
        #log(INFO, f"Client instruction : Len {len(client_instructions)} {client_instructions[0][1].config} {client_instructions[1][1].config}")

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fl.server.server.fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
    
        
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()
        
        # Early Stopping
        min_val_loss = float("inf")
        max_metric_dis = 0.
        round_no_improve = 0
        path = Path(self.path_log, "rounds_time.csv")
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            write_time_csv(path, current_round, "fit", "start fit call")
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            write_time_csv(path, current_round, "fit", "res fit received")
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
            write_time_csv(path, current_round, "fit", "res fit aggregated")

            # Evaluate model using strategy implementation
            write_time_csv(path, current_round, "evaluate", "central evaluate call")
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            write_time_csv(path, current_round, "evaluate", "central evaluated")
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
            write_time_csv(path, current_round, "evaluate", "central evaluate end")

            # Evaluate model on a sample of available clients
            write_time_csv(path, current_round, "evaluate", "distributed evaluate call")
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            write_time_csv(path, current_round, "evaluate", "distributed evaluated")
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                key_metrics = list(evaluate_metrics_fed.keys())
                acc = evaluate_metrics_fed[key_metrics[0]]
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                    write_time_csv(path, current_round, "evaluate", "distributed evaluate end")
                    # Early Stopping
                    if acc > 0.75 + 1e-6:
                        log(INFO, "EARLY STOPPING, ACCURACY > 0.75")
                        break
                    else:
                        if current_round >= 100: # Start Early Stopping after 100 rounds
                            if acc > max_metric_dis:
                                round_no_improve = 0
                                max_metric_dis = acc
                            else:
                                round_no_improve += 1
                                if round_no_improve == self.wait_round:
                                    log(INFO, "EARLY STOPPING")
                                    break
                        # if loss_fed < min_val_loss:
                        #     round_no_improve = 0
                        #     min_val_loss = loss_fed
                        # else:
                        #     round_no_improve += 1
                        #     if round_no_improve == self.wait_round:
                        #         log(INFO, "EARLY STOPPING")
                        #         break
                    
        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history        


@hydra.main(config_path="config", config_name="config_file",version_base=None)
def main(cfg:DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    server_address = cfg.comm.host
    server_port = cfg.comm.port
    output_dir = HydraConfig.get().runtime.output_dir
    logging.info(output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"USING DEVICE : {device}")    
    # Get initial parameters
    model = instantiate(cfg.neuralnet)
    model = convert_bn_to_gn(model, num_groups=cfg.params.num_groups)
    print(cfg.optimizer)
    if 'constraints' in cfg.keys():
        logging.info("Applying constraints")
        constraints = instantiate(cfg.constraints, model=model)
        make_feasible(model,constraints)
        model_parameters = get_parameters(model)
        initial_parameters = ndarrays_to_parameters(model_parameters)
    else:
        logging.info("No constraints specified")
        model_parameters = get_parameters(model)
        initial_parameters = ndarrays_to_parameters(model_parameters)
    
    
    #Load TestData
    testloader = load_testdata_from_file(cfg.data)
    
    strategy = instantiate(cfg.strategy,
                           initial_parameters=initial_parameters,
                           on_fit_config_fn=get_on_fit_config(cfg.client),
                           evaluate_metrics_aggregation_fn=weighted_average,
                           evaluate_fn=get_evaluate_fn(model, testloader,device,cfg.params,output_dir),
                           on_evaluate_config_fn=get_on_evaluate_config(cfg.client)
                        )
    
    
    print(strategy.__repr__())
    customserver = CustomServer(wait_round=cfg.params.wait_round, 
                                path_log = output_dir,
                                #client_manager=CustomSimpleClientManager(), #change to SimpleClientManagerCustom for 100 clients
                                client_manager = fl.server.SimpleClientManager(),
                                strategy=strategy,
                                )
    
    hist = fl.server.start_server(
        server_address=str(server_address)+":"+str(server_port),
        server=customserver,
        config=fl.server.ServerConfig(num_rounds=cfg.params.num_rounds),
        strategy=strategy
        )
    
    results_path = Path(output_dir)/"results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(hist, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return output_dir


if __name__ == "__main__":
        
    logging.basicConfig(
        # filename=args.log_dir + args.log_file, 
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d : %(message)s',
        )
    
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of available GPUs: {num_gpus}")

        # Get information about each GPU
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            logging.info(f"GPU {i + 1}: {gpu_name}")
    else:
        logging.info("CUDA (GPU support) is not available on this system.")
    
    try:
        main()
    except Exception as err:
        logging.error(err)
    
    
    