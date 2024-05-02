
from omegaconf import DictConfig
import torch
from collections import OrderedDict
from model import SimpleNet, test, Net
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from flwr.server import Server, History
from flwr.common.logger import log
from logging import DEBUG, INFO
import flwr as fl 
from typing import Optional
import timeit

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round):
        return {'lr': config.lr, 'local_epochs': config.local_epochs}
    return fit_config_fn

def set_parameters(model, parameters)->None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
    model.load_state_dict(state_dict, strict = True)
    return model

def get_evaluate_config(model, testloader):
    def evaluate_fn(server_round, parameters, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        set_parameters(model, parameters)
        loss, accuracy = test(model, testloader, verbose=False) 
        return loss, {"accuracy": accuracy}
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

    

