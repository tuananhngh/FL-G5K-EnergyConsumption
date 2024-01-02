import server
import hydra
import flwr as fl
import utils
import torch
import pickle 
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters

@hydra.main(config_path="config", config_name="server_config")
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    server_address = cfg.comm.server_address
    server_port = cfg.comm.server_port
    output_dir = HydraConfig.get().runtime.output_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = utils.Net(num_classes=10)
    model_parameters = utils.get_parameters(model)
    initial_parameters = ndarrays_to_parameters(model_parameters)
    _,_,testloader = utils.load_dataset(cfg.client_params)
    #_,_, testloader = utils.load_dataloader(1, cfg.params.path_to_data)
    
    strategy = fl.server.strategy.FedAvg(
        initial_parameters=initial_parameters,
        evaluate_fn=server.get_evaluate_fn(testloader, cfg.params),
        evaluate_metrics_aggregation_fn=server.weighted_average,
        on_fit_config_fn=server.get_on_fit_config(cfg.client_params),
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = 2,
        min_evaluate_clients =2,
        min_available_clients=2        
    )
    
    hist = fl.server.start_server(
        server_address=str(server_address)+":"+str(server_port),
        config=fl.server.ServerConfig(num_rounds=cfg.params.num_rounds),
        strategy=strategy,
    )
    results_path = Path(output_dir)/"results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(hist, f, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    main()
    
    