from comm_utils import load_server_data, load_client_data, process_rounds_time, filter_round_time, process_host_round_time, read_server_clients_data, plot_multples_clients, plot_send_receive_evolution_round
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
from box import Box
import json
import sys
from filter_exp import read_summaryfile, create_json_file
from result_utils import read_server
import seaborn as sns
sns.set_theme(style="whitegrid")

client_list = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5', 'client_6', 'client_7', 'client_8', 'client_9']
strategies = ['fedsfw', 'fedadam','fedadagrad','fedyogi']
usr_homedir = "/User/Slaton/Documents/grenoble-code/fl-flower"
parent_path = "/Users/Slaton/Documents/grenoble-code/fl-flower/energyfl/outputcifar10/10clients/comm/"

match_condition=lambda summary: (
        (summary["client.local_epochs"].isin([1,3,5])) & 
        (((summary["strategy"].isin(["fedadam","fedadagrad","fedyogi"])) & (summary["client.lr"]==0.0316)) |
        ((summary["strategy"] == 'fedsfw') & (summary["client.lr"]==0.0316)))
    )

strategies_dict = create_json_file(strategies, parent_path , usr_homedir, match_condition)