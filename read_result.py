import pickle as pkl
from flwr.server.history import History
import matplotlib.pyplot as plt
path = "./outputs/2023-12-18/sample10/results.pkl"
path2 = "./outputs/2023-12-18/sample50/results.pkl"

with open(path2, "rb") as f:
    result = pkl.load(f)
    
acc = [val for (it,val) in result.metrics_centralized["accuracy"]]
plt.plot(acc)