import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

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


def read_data_from_dict(strat_dict, strategy, epoch, exp, host, file_type):
    """
    Read data from a dictionary containing paths to data files.
    """
    path = strat_dict[strategy]['split_epoch'][epoch][exp][host][file_type]
    return path

def concat_server_results(strategy, epoch, host, file_type, strategies_dict, n_exp = 5):
    """
    Concatenate server results from multiple experiments.
    """
    result_paths = [read_data_from_dict(strategies_dict, strategy, epoch, f'exp_{i}' , host, file_type) for i in range(n_exp)]
    result_dfs = []
    for path in result_paths:
        result_dfs.append(read_server(path))
    results_df = pd.concat(result_dfs, axis=1, keys=[f'exp_{i}' for i in range(n_exp)])
    return results_df

def average_results(results_df, column, n_exp=5):
    """
    Average results from multiple experiments.
    """
    concat = pd.concat([results_df.__getattr__(f'exp_{i}')[column] for i in range(n_exp)],axis=1)
    avg_df = concat.mean(axis=1)
    std_df = concat.std(axis=1)
    lower = avg_df - std_df
    upper = avg_df + std_df
    final_df = pd.DataFrame({f"{column}_avg": avg_df, f"{column}_std": std_df, f"{column}_lower_bound": lower, f"{column}_upper_bound": upper})
    return final_df


def compute_strategy_averages_results(strategies, strategies_dict, epoch, metric, n_exp):
    stratavg = []
    for strat in strategies:
        strat_concat = concat_server_results(strat, epoch, 'server', 'results', strategies_dict, n_exp)
        strat_avg = average_results(strat_concat, metric, n_exp)
        stratavg.append(strat_avg)
    final_data = pd.concat(stratavg, axis=1, keys=strategies)
    return final_data


def plot_results(strategies, strategies_dict, epoch, metric, n_exp):
    """
    Plot results from multiple experiments.
    """
    data = compute_strategy_averages_results(strategies, strategies_dict, epoch, metric, n_exp)
    fig, axs = plt.subplots(1, figsize=(20, 5))
    for strat in strategies:
        axs.plot(data[strat][f'{metric}_avg'], label=strat)
        #axs.fill_between(data[strat].index, data[strat][f'{metric}_lower_bound'], data[strat][f'{metric}_upper_bound'], alpha=0.2)
    fig.legend(strategies, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(strategies), fontsize=12)
    axs.set_xlabel('Server Round')
    axs.set_ylabel(metric)
    plt.show()