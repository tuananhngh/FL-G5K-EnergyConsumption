import pickle as pkl
from flwr.server.history import History
import matplotlib.pyplot as plt
import os
from typing import Dict



def result_one_exp(path_to_folder:str, metric:str)->None:
    result_file = "results.pkl"
    path_to_result = os.path.join(path_to_folder, result_file)
    if os.path.exists(path_to_result):
        with open(path_to_result, "rb") as f:
            result = pkl.load(f)
        metric_centralized = [val for (it,val) in result.metrics_centralized[metric]]
        metric_distributed = [val for (it,val) in result.metrics_distributed[metric]]
        
        loss_centralized = [val for (it,val) in result.losses_centralized]
        loss_distributed = [val for (it,val) in result.losses_distributed]
        return metric_distributed, loss_distributed #type:ignore
    else:
        raise FileNotFoundError(f"File {result_file} not found in {path_to_folder}")

def result_multi_exp(path_to_multirun:str, metric:str)->Dict:
    ls_folder = os.listdir(path_to_multirun)
    ls_folder = [os.path.join(path_to_multirun, folder) for folder in ls_folder if os.path.isdir(os.path.join(path_to_multirun, folder))]
    ls_folder = sorted(ls_folder)
    result_val = {}
    for folder in ls_folder:
        folder_idx = folder.split("/")[-1]
        print(folder_idx)
        try:
            metric_distributed, loss_distributed = result_one_exp(folder, metric) #type:ignore
            result_val[folder_idx] = (metric_distributed, loss_distributed)
        except FileNotFoundError as e:
            print(e)
    return result_val #type:ignore

def plot_result(result:dict, metric:str)->None:
    avg_round = 30
    fig, ax = plt.subplots(1, 2, figsize=(20,5))
    for key, val in result.items():
        print(key)
        metric_dis, loss_dis = val[0],val[1]
        metric_dis_avg, loss_dis_avg = sum(metric_dis[-avg_round:])/avg_round, sum(loss_dis[-avg_round:])/avg_round
        ax[0].plot(metric_dis, label=f"Avg Acc: {metric_dis_avg:.4f}", linestyle="-")
        ax[1].plot(loss_dis, label=f"Avg Loss :{loss_dis_avg:.4f}", linestyle="-")
    ax[0].set_xlabel("Round")
    ax[0].set_ylabel(metric)
    ax[1].set_ylabel("Loss")
    legend = fig.legend(result.keys(), 
                        loc="upper center", 
                        bbox_to_anchor=(0.5, 1.0), 
                        ncol=len(result.keys()),
                        fontsize=12,)
    ax[0].legend()
    ax[1].legend()
    plt.show()


if __name__=="__main__":
    multi_run_folder = "/home/tunguyen/energyfl/Simulation/cifar10/sfw/multirun/2024-04-29 21-31-30"
    vals = result_multi_exp(multi_run_folder, "accuracy")
    plot_result(vals, "accuracy")