from utils.training import get_parameters, test, seed_everything, set_parameters
from utils.datahandler import load_testdata_from_file
import hydra
import numpy as np
import flwr as fl
import logging
import torch
import pickle 
import timeit
from collections import OrderedDict
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from flwr.common import Metrics, NDArray, Scalar, ndarrays_to_parameters, FitIns, EvaluateRes
from hydra.utils import instantiate
from typing import Callable, Dict, Tuple, List, Optional
from flwr.server.history import History
from logging import DEBUG, INFO
from flwr.common.logger import log
from utils.models import convert_bn_to_gn
from flwr.server.strategy import FedAvg, FedYogi, FedAdam

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


def get_evaluate_fn(model, testloader, device, cfg: Dict[str, Scalar])->Callable:
    def evaluate_fn(server_round:int, parameters:NDArray, config):
        model.to(device)
        set_parameters(model, parameters)
        loss, accuracy = test(model, testloader, device, verbose=False)
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate_fn

class CustomServer(fl.server.Server):
    def __init__(self,wait_round:int,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_round = wait_round
        
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
        round_no_improve = 0
        
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
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


            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                    # Early Stopping
                    if current_round >= 100: # Start Early Stopping after 100 rounds
                        if loss_fed < min_val_loss:
                            round_no_improve = 0
                            min_val_loss = loss_fed
                        else:
                            round_no_improve += 1
                            if round_no_improve == self.wait_round:
                                log(INFO, "EARLY STOPPING")
                                break
                    
        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history        


@hydra.main(config_path="config", config_name="config_file")
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
    model_parameters = get_parameters(model)
    initial_parameters = ndarrays_to_parameters(model_parameters)
    
    #Load TestData
    testloader = load_testdata_from_file(cfg.data)
    
    strategy = instantiate(cfg.strategy,
                           initial_parameters=initial_parameters,
                           on_fit_config_fn=get_on_fit_config(cfg.client),
                           evaluate_metrics_aggregation_fn=weighted_average,
                           evaluate_fn=get_evaluate_fn(model, testloader,device,cfg.params),
                           on_evaluate_config_fn=get_on_evaluate_config(cfg.client)
                           )
    
    # strategy = FedAdam(initial_parameters=initial_parameters,
    #                     fraction_fit=cfg.params.fraction_fit,
    #                     fraction_evaluate=cfg.params.fraction_evaluate,
    #                     on_evaluate_config_fn=get_on_evaluate_config(cfg.client),
    #                     on_fit_config_fn=get_on_fit_config(cfg.client),
    #                     evaluate_metrics_aggregation_fn=weighted_average,
    #                     evaluate_fn=get_evaluate_fn(model, testloader,device,cfg.client),
    #                     min_fit_clients=cfg.params.num_clients_per_round_fit,
    #                     min_evaluate_clients=cfg.params.num_clients_per_round_eval,
    #                     eta=cfg.params.lr,
    #                     eta_l=cfg.client.lr,
    #                     tau=0.001)
    
    print(strategy.__repr__())
    customserver = CustomServer(wait_round=cfg.params.wait_round, 
                                client_manager=fl.server.SimpleClientManager(), 
                                strategy=strategy)
    
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
    
    
    