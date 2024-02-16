import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Callable, Dict
from fedrecon import FedRecon
from matrix_factorization import build_reconstruction_model
from load_movielens import load_movielens_data,path_to_1m
from flwr.common import (
    Metrics,
    FitIns,
    FitRes,
    Scalar
)
import os
import pickle as pkl

def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss": sum(accuracies) / sum(examples)}

def get_on_fit_config(config: Dict[str, Scalar])->Callable:
    def fit_config_fn(server_round:int)->FitIns:
        return {'recon_epochs': 3, 'recon_lr':0.01, 'pers_epochs': 3, 'pers_lr':0.01 }
    return fit_config_fn

def main():
    ratings_df, movies_df = load_movielens_data(path_to_1m)
    num_users, num_items = len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
    model, global_params, local_params = build_reconstruction_model(num_users=1, num_items=num_items, num_latent_factors=50, personal_model=True, add_biases=False, l2_regularizer=0.0, spreadout_lambda=0.0)
    
    hist = fl.server.start_server(server_address="[::]:8080",
                            config=fl.server.ServerConfig(num_rounds=5), 
                            strategy=FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
                                            on_fit_config_fn=get_on_fit_config,
                                            initial_parameters=global_params) 
                            )
    result_path = os.getcwd() + "/result.pkl"
    with open(result_path, 'wb') as f:
        pkl.dump(hist, f)
        
if __name__ == "__main__":  
    main()

