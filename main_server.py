import csv
from tracemalloc import start
import server
import hydra
import flwr as fl
import logging
import utils
import torch
import pickle 
import models
import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from hydra.utils import instantiate


@hydra.main(config_path="config", config_name="config_file")
def main(cfg:DictConfig):
    logging.info(OmegaConf.to_yaml(cfg))
    server_address = cfg.comm.host
    server_port = cfg.comm.port
    output_dir = HydraConfig.get().runtime.output_dir
    logging.info(output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Device: ", device)
    
    # Get initial parameters
    model = instantiate(cfg.neuralnet)
    #model = models.Net()
    #model.to(device)
    model_parameters = utils.get_parameters(model)
    initial_parameters = ndarrays_to_parameters(model_parameters)
    
    #Load TestData
    _,_,testloader = utils.load_dataset(cfg.params)
    #_,_, testloader = utils.load_dataloader(1, cfg.params.path_to_data)
    
    strategy = instantiate(cfg.strategy,
                           initial_parameters=initial_parameters,
                           on_fit_config_fn=server.get_on_fit_config(cfg.client),
                           evaluate_metrics_aggregation_fn=server.weighted_average,
                           evaluate_fn=server.get_evaluate_fn(model, testloader,device,cfg.params),
                           on_evaluate_config_fn=server.get_on_evaluate_config(cfg.client)
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
    
    
    