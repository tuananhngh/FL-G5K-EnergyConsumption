import pickle as pkl
from flwr.server.history import History
import matplotlib.pyplot as plt
import os

def foo(pathadam:str):
    with open(pathadam, "rb") as f:
        resultadam = pkl.load(f)
    # with open(pathavg, "rb") as f:
    #     resultavg = pkl.load(f)
    # with open(pathyogi, "rb") as f:
    #     resultyogi = pkl.load(f)

    acc_centralized_adam= [val for (it,val) in resultadam.metrics_centralized["accuracy"]]
    acc_distributed_adam = [val for (it,val) in resultadam.metrics_distributed["accuracy"]]
    
    loss_centralized_adam= [val for (it,val) in resultadam.losses_centralized]
    loss_distributed_adam = [val for (it,val) in resultadam.losses_distributed]
    
    # acc_distributed_avg = [val for (it,val) in resultavg.metrics_distributed["accuracy"]]
    # acc_centralized_avg = [val for (it,val) in resultavg.metrics_centralized["accuracy"]]
    
    # acc_centralized_yogi = [val for (it,val) in resultyogi.metrics_centralized["accuracy"]]
    # acc_distributed_yogi = [val for (it,val) in resultyogi.metrics_distributed["accuracy"]]
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(acc_centralized_adam, label="FedSparseFW Centralized",)
    ax[0].plot(acc_distributed_adam, label="FedSparseFW Distributed", linestyle="--")
    ax[1].plot(loss_centralized_adam, label="FedSparseFW Centralized",)
    ax[1].plot(loss_distributed_adam, label="FedSparseFW Distributed", linestyle="--")
    # ax.plot(acc_centralized_avg, label="FedAvg Centralized",)
    # ax.plot(acc_distributed_avg, label="FedAvg Distributed", linestyle="--")
    # ax.plot(acc_centralized_yogi, label="FedYogi Centralized",)
    # ax.plot(acc_distributed_yogi, label="FedYogi Distributed", linestyle="--")
    ax[0].set_xlabel("Round")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    
    ax[1].set_xlabel("Round")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    
if __name__ == "__main__":
    #path_adam = "/home/tunguyen/federated-learning/simulation-flwr/outputs/2024-02-23/15-12-40"
    #path_adam = "/home/tunguyen/energyfl/Simulation/cifar10/sfw/multirun/2024-04-26 00-39-43/config_fit.lr=0.1"
    path_adam = "/home/tunguyen/energyfl/Simulation/cifar10/sfw/2024-04-27 22-18-47"
    
    #path_avg = "/home/tunguyen/federated-learning/simulation-flwr/outputs/2024-02-23/10-47-55"
    #path_avg = "/home/tunguyen/federated-learning/simulation-flwr/outputs/2024-02-24/11-51-41"
    # path_avg = "/home/tunguyen/federated-learning/simulation-flwr/outputs/2024-02-25/13-18-38"
    
    # #path_yogi = "/home/tunguyen/federated-learning/simulation-flwr/outputs/2024-02-23/13-16-22"
    # path_yogi = "/home/tunguyen/federated-learning/simulation-flwr/outputs/2024-02-24/14-43-48"
    
    result = "results.pkl"
    #path_to_result_avg = os.path.join(path_avg, result)
    path_to_result_adam = os.path.join(path_adam, result)
    #path_to_result_yogi = os.path.join(path_yogi, result)
    foo(path_to_result_adam)
    
#     return result
    
# acc = [val for (it,val) in result.metrics_centralized["accuracy"]]
# plt.plot(acc)