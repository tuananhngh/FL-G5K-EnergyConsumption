from utils.training import get_parameters, test, seed_everything
from utils.datahandler import load_testdata_from_file
import hydra
import flwr as fl
import logging
import torch
import pickle 
from collections import OrderedDict
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from flwr.common import Scalar, ndarrays_to_parameters
from flwr.common import Metrics, NDArray, Scalar, ndarrays_to_parameters, FitIns, EvaluateRes
from hydra.utils import instantiate
from typing import Callable, Dict, Tuple, List

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
        num_classes = cfg["num_classes"]
        steps = len(testloader)
        params_dict = zip(model.state_dict().keys(),parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        #utils.set_parameters(model, parameters)
        loss, accuracy = test(model, testloader, device, verbose=False)
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate_fn


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
    model_parameters = get_parameters(model)
    initial_parameters = ndarrays_to_parameters(model_parameters)
    
    #Load TestData
    #dataconfig = DataSetHandler(cfg.data)
    #_,_,testloader = dataconfig()
    #_,_, testloader = utils.load_dataloader(1, cfg.params.path_to_data)
    testloader = load_testdata_from_file(cfg.data)
    
    strategy = instantiate(cfg.strategy,
                           initial_parameters=initial_parameters,
                           on_fit_config_fn=get_on_fit_config(cfg.client),
                           evaluate_metrics_aggregation_fn=weighted_average,
                           evaluate_fn=get_evaluate_fn(model, testloader,device,cfg.params),
                           on_evaluate_config_fn=get_on_evaluate_config(cfg.client)
                           )
    
    hist = fl.server.start_server(
        server_address=str(server_address)+":"+str(server_port),
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
    
    
    