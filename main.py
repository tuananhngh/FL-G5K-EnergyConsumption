import pickle
from pathlib import Path
import flwr as fl
import hydra
import torch
from typing import List, Tuple
from hydra.core.hydra_config import HydraConfig
from model import SimpleNet, Net
from omegaconf import OmegaConf, DictConfig
from dataset import load_dataset
from client import generate_client
from server import get_evaluate_config
from server import get_on_fit_config, set_parameters
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models import efficientnet, AlexNet

def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

@hydra.main(config_path="", config_name="base", version_base=None)
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    #device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path = HydraConfig.get().runtime.output_dir
    # Get Dataloaders 
    trainloaders, valloaders, testloader = load_dataset(cfg.path_to_data, cfg.num_clients, cfg.batch_size)
    print(f"#Clients : {cfg.num_clients}, #Data/Clients : {len(trainloaders[0].dataset)}")
    # Client generator
    client_fn = generate_client(trainloaders, valloaders)
    model  = mobilenet_v3_small(num_classes=10)
    #initial_params = [val.cpu().numpy() for name,val in model.state_dict().items() if 'bn' not in name]
    initial_params = [val.cpu().numpy() for name,val in model.state_dict().items() if 'num_batches_tracked' not in name]
    initial_params = ndarrays_to_parameters(initial_params)
    # Strategy
    strategy = fl.server.strategy.FedAdam(
        fraction_fit=0.2,
        min_fit_clients = cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        #min_available_clients = cfg.num_clients,
        initial_parameters=initial_params,
        fraction_evaluate=0.2,
        on_fit_config_fn = get_on_fit_config(cfg.config_fit),
        evaluate_fn = get_evaluate_config(model, testloader),
        evaluate_metrics_aggregation_fn=weighted_average,
        eta=cfg.config_fit.server_lr,
        eta_l=cfg.config_fit.lr,
    )
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients = cfg.num_clients,
        config = fl.server.ServerConfig(
            num_rounds = cfg.num_rounds,
        ),
        strategy = strategy,
        client_resources={"num_cpus": 4, "num_gpus":0},
    )
    rerults_path = Path(save_path) / "results.pkl"
    with open(rerults_path, "wb") as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()