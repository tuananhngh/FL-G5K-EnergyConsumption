import pickle
from pathlib import Path
import random
import flwr as fl
import hydra
import torch
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import List, Tuple
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from model import SimpleNet, Net, convert_bn_to_gn, check_device
from omegaconf import OmegaConf, DictConfig
from dataset import load_dataset
from client_sparse import generate_client
from server import get_evaluate_config
from server import get_on_fit_config, set_parameters, CustomServer, get_evaluate_config
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server.strategy import FedAvg, FedAdam, FedYogi, FedAdagrad, FedMedian
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models import efficientnet, AlexNet, resnet18
from strategy import fedsfw
from optimizers.ConstraintsSet import create_lp_constraints,create_k_sparse_constraints,make_feasible
from serialization import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use it in your code
seed = 42
seed_everything(seed)

def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": np.sum(accuracies) / np.sum(examples)}

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = HydraConfig.get().runtime.output_dir
    # Get Dataloaders 
    trainloaders, valloaders, testloader = load_dataset(cfg.path_to_data, cfg.num_clients, cfg.batch_size, cfg.iid)
    print(f"#Clients : {cfg.num_clients}, #Data/Clients : {len(trainloaders[0].dataset)}")

    #model  = resnet18(num_classes=10)
    model = resnet18(pretrained=False, num_classes=10)
    model = convert_bn_to_gn(model, num_groups=cfg.num_groups)
    model.to(device)
    #model = torch.nn.DataParallel(model,device_ids=[0,1])
    check_device(model)
    
    #Create constraints
    #constraints = create_lp_constraints(model, ord=1, mode='initialization')
    #constraints = create_k_sparse_constraints(model, K=300, K_frac=0.2, mode='radius')
    constraints = instantiate(cfg.constraints, model=model)
    make_feasible(model, constraints)
    
    initial_params = [val.cpu().numpy() for name,val in model.state_dict().items()]
    initial_params = ndarrays_to_sparse_parameters(initial_params)
    
    # Client generator
    client_fn = generate_client(model, constraints, trainloaders, valloaders)
    
    # Strategy
    # strategy = instantiate(cfg.strategy,
    #                        fraction_fit = 1,
    #                        fraction_evaluate=1,
    #                        min_fit_clients=cfg.num_clients_per_round_fit,
    #                        min_evaluate_clients=cfg.num_clients_per_round_eval,
    #                        #initial_parameters=initial_params,
    #                        on_fit_config_fn = get_on_fit_config(cfg.config_fit),
    #                        evaluate_fn = get_evaluate_config(model, testloader),
    #                        evaluate_metrics_aggregation_fn=weighted_average,)
    
    
    
    strategy = instantiate(cfg.strategy,
                            fraction_fit = 1,
                            fraction_evaluate=1,
                            initial_parameters=initial_params,
                            min_fit_clients=cfg.num_clients_per_round_fit,
                            min_evaluate_clients=cfg.num_clients_per_round_eval,
                            on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                            evaluate_fn=get_evaluate_config(model, testloader),
                            evaluate_metrics_aggregation_fn=weighted_average,
                            info_path=save_path
    )
    
    
    customserver = CustomServer(wait_round=cfg.wait_round, 
                            client_manager=fl.server.SimpleClientManager(), 
                            strategy=strategy)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients = cfg.num_clients,
        config = fl.server.ServerConfig(
            num_rounds = cfg.num_rounds,
        ),
        server = customserver,
        #strategy = strategy,
        ray_init_args = {'num_cpus': 32, 'num_gpus': 4},
        client_resources={"num_cpus": 4, "num_gpus":0.1},
    )
    rerults_path = Path(save_path) / "results.pkl"
    with open(rerults_path, "wb") as f:
        pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
        

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