import pickle as pkl
import pandas as pd

def flwr_pkl(path_to_pkl):
    """
    Load and return the contents of a pickle file.

    Parameters:
    path_to_pkl (str): The path to the pickle file.

    Returns:
    object: The deserialized object from the pickle file.
    """
    with open(path_to_pkl, "rb") as f:
        result = pkl.load(f)
    return result

def read_server(path_to_result):
    """_summary_
    
    Returns:
        SimpleNamespace[pd.DataFrame]: Contains energy, results as DataFrames
    """
    try :
        results = flwr_pkl(path_to_result)
    except FileNotFoundError as err:
        print(err)
        results = None
        results_df = None
    if results is not None:
        acc_centralized = [acc[1] for acc in results.metrics_centralized["accuracy"][1:]]
        acc_distributed = [acc[1] for acc in results.metrics_distributed["accuracy"]]
        losses_centralized = [loss[1] for loss in results.losses_centralized[1:]] # First loss is evaluated on initial parameters
        losses_distributed = [loss[1] for loss in results.losses_distributed]
        server_round = [i for i in range(1,len(acc_centralized)+1)]
        #print(len(server_round), len(acc_centralized), len(acc_distributed), len(losses_centralized), len(losses_distributed))
        results_df = pd.DataFrame(
            {
                "server_round": server_round,
                "acc_centralized": acc_centralized,
                "acc_distributed": acc_distributed,
                "losses_centralized": losses_centralized,
                "losses_distributed": losses_distributed   
            }
        )
    return results_df