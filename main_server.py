import server
import hydra
import flwr as fl
import utils
import torch
import pickle 
import models
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters


# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get information about each GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i + 1}: {gpu_name}")
else:
    print("CUDA (GPU support) is not available on this system.")


@hydra.main(config_path="config", config_name="config_file")
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    server_address = cfg.comm.host
    server_port = cfg.comm.port
    output_dir = HydraConfig.get().runtime.output_dir
    print(output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
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
                           )
    
    hist = fl.server.start_server(
        server_address=str(server_address)+":"+str(server_port),
        config=fl.server.ServerConfig(num_rounds=cfg.params.num_rounds),
        strategy=strategy
    )
    results_path = Path(output_dir)/"results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(hist, f, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    main()
    
    
    