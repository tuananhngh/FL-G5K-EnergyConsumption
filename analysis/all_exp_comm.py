# Standard library imports
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style
from box import Box

theme_dict = {**axes_style("whitegrid"), "grid.linestyle": ":"}
so.Plot.config.theme.update(theme_dict)

# Local application imports
from comm_utils import (load_server_data, load_client_data, process_rounds_time, 
                        filter_round_time, process_host_round_time, read_server_clients_data, 
                        plot_multples_clients, plot_send_receive_evolution_round, 
                        process_network_data, melt_send_receive_df, sum_send_receive)

from filter_exp import read_summaryfile, create_json_file
from result_utils import read_server

# Set theme for seaborn
sns.set_theme(style="whitegrid")

client_list = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5', 'client_6', 'client_7', 'client_8', 'client_9']
strategies = ['fedadam','fedyogi', 'fedavg', 'fedadagrad']
usr_homedir = "/Users/Slaton/Documents/grenoble-code/fl-flower"
parent_path = "/Users/Slaton/Documents/grenoble-code/fl-flower/energyfl/outputcifar10/10clients/"

#filter round time at 75%


match_condition=lambda summary: (
        (summary["client.local_epochs"].isin([1,3,5])) & 
        (
            ((summary["strategy"].isin(["fedadam","fedyogi","fedavg","fedadagrad"])) & (summary["client.lr"]==0.0316))
            # ((summary["strategy"] == 'fedsfw') & (summary["client.lr"]==0.01) & (summary["sparse_constraints.K_frac"]==0.1)) |
            # ((summary["strategy"] == 'fedconstraints') & (summary["client.lr"]==0.0316) & (summary["sparse_constraints.sparse_prop"]==0.5) & (summary["optimizer"]=="SFW")
            #  )
        )
    )

strategies_dict = Box(create_json_file(strategies, parent_path , usr_homedir, match_condition))


def get_bandwidth(strategy:str, epoch:str, strategies_dict:Box):
    summary_info = strategies_dict.__getattr__(strategy).split_epoch.__getattr__(epoch).exp_0.summary
    exp_path = summary_info.result_folder
    exp_duration = summary_info.timestamps_end_experiment - summary_info.timestamps_start_experiment
    files = read_server_clients_data(exp_path)
    server_round_time = process_rounds_time(files.server.time)
    
    send_df, receive_df = process_network_data(client_list, files)
    df_send_receive = melt_send_receive_df(send_df, receive_df)
    df_send_receive_sum = sum_send_receive(send_df, receive_df)
    return df_send_receive, df_send_receive_sum, server_round_time, exp_duration

#ok = get_bandwidth('fedavg', 'epoch_1', strategies_dict)

def get_bandwidth_server(strategy:str, epoch:str, strategies_dict:Box):
    summary_info = strategies_dict.__getattr__(strategy).split_epoch.__getattr__(epoch).exp_0.summary
    exp_path = summary_info.result_folder
    exp_duration = summary_info.timestamps_end_experiment - summary_info.timestamps_start_experiment
    files = read_server_clients_data(exp_path)
    server_round_time = process_rounds_time(files.server.time)

    server_send_df, server_receive_df = process_network_data(['server'], files)
    df_server_send_receive = melt_send_receive_df(server_send_df, server_receive_df)
    return df_server_send_receive, server_round_time, exp_duration

def melt_strategies_server(strategies, epoch:str, strategies_dict:Box):
    strats_round_time = []
    strats_send_receive = []
    exps_dur = {}
    for i,strat in enumerate(strategies):
        send_receive, server, exp_dur = get_bandwidth_server(strat, epoch, strategies_dict)
        server['strategy'] = strat
        send_receive['strategy'] = strat
        exps_dur[strat] = exp_dur
        strats_round_time.append(server)
        strats_send_receive.append(send_receive)
    total_roundtime = pd.concat(strats_round_time)
    total_send_receive = pd.concat(strats_send_receive)
    exps_dur_df = pd.DataFrame.from_dict(exps_dur, orient='index', columns=['exp_duration'])
    exps_dur_df.reset_index(inplace=True)
    exps_dur_df.rename(columns={'index':'strategy'}, inplace=True)
    exps_dur_df['exp_duration'] = exps_dur_df['exp_duration'].dt.total_seconds()/3600
    return total_send_receive, total_roundtime, exps_dur_df



def plot_bandwidth_clients(strategies, epoch:str, strategies_dict:Box):
    strats_round_time = []
    strats_duration = []
    strats_sum = []
    fig, ax = plt.subplots(len(strategies), figsize=(20,5*len(strategies)))
    for i, strat in enumerate(strategies):
        strat_sr, strat_sr_sum, round_time, duration = get_bandwidth(strat, epoch, strategies_dict)
        strats_round_time.append(round_time)
        strats_duration.append(duration)
        strat_sr_sum['strategy'] = strat
        strats_sum.append(strat_sr_sum)
        sns.boxplot(x='client', hue='status',y='value', data=strat_sr, ax=ax[i])
        ax[i].set_ylabel('Total Bandwidth/Round (MB)')
        ax[i].set_title(strat)
        ax[i].legend_.remove()
        ax[i].set_xlabel(' ')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.01), fontsize=12)
    plt.tight_layout()
    plt.show()
    
    fig_sum, ax_sum = plt.subplots(1, figsize=(20,5))
    sns.barplot(x='strategy', y='value', hue='status', data=pd.concat(strats_sum))
    plt.show()
    
def plot_bandwidth_server(strategies, epoch, strategies_dict):
    total_send_receive, total_roundtime, exps_dur_df = melt_strategies_server(strategies, epoch, strategies_dict)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    sns.boxplot(data=total_send_receive, x='strategy', y='value', hue='status', ax=ax[0])
    ax[0].set_title('Total Send/Receive')
    ax[0].set_ylabel('Total Bandwidth/Round (MB)')
    ax[0].legend(loc='upper center', ncol=2, bbox_to_anchor=(0.225, 1.01))

    sns.barplot(data=total_roundtime, x='strategy', y='round_duration', ax=ax[1])
    ax[1].set_title('Total Round Time')
    ax[1].set_ylabel('Round Time (s)')
    ax[1].set_xlabel('Strategy')

    sns.barplot(data=exps_dur_df, x='strategy', y='exp_duration', ax=ax[2])
    ax[2].set_title('Experiment Duration')
    ax[2].set_ylabel('Experiment Duration (h)')
    plt.tight_layout()
    plt.show()
    
#plot_bandwidth_server(strategies, 'epoch_1', strategies_dict)
#plot_bandwidth_clients(strategies, 'epoch_1', strategies_dict)

from result_utils import compute_strategy_averages_results, plot_results
results_df = compute_strategy_averages_results(strategies, strategies_dict, 'epoch_1', 'acc_centralized', 5)
#plot_results(strategies, strategies_dict, 'epoch_1', 'acc_centralized',1)