from process_data import (process_rounds_time, 
                          filter_round_time,
                          melt_strategies_server,
                          get_bandwidth,
)
from box import Box
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_message_size_client(df:pd.DataFrame, message_type:str='send'):
    """
    This function plots a boxplot of the average message size for each host.
    Args:
        df (pd.DataFrame): The DataFrame containing the message size data for each round. 
                         where each row is a message size for a specific round and column is the host.
    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title(f'Average {message_type} message size')
    plt.xlabel('Host')
    plt.ylabel('MB')
    
    
def plot_multples_clients(file_holder):
    server_time = file_holder.server.time
    clients = sorted([client for client in file_holder.keys() if 'client' in client])
    server_round_time = process_rounds_time(server_time, mode='round')
    server_filter = filter_round_time(server_round_time, file_holder.server.network)
    
    fig, axs = plt.subplots(2, 1, figsize=(20, 5))
    rounds = server_filter['round'].unique()
    for client in clients:
        client_network = file_holder[client].network
        client_filtered = filter_round_time(server_round_time, client_network)
        sends, receives = [], []
        for r in rounds:
            send_sum = client_filtered[client_filtered['round'] == r].send.sum()/1024
            recv_sum = client_filtered[client_filtered['round'] == r].receive.sum()/1024
            sends.append(send_sum)
            receives.append(recv_sum)
        axs[0].plot(rounds, sends, label=f"{client}")#,marker='^', markevery=1, linewidth=2)
        axs[1].plot(rounds, receives, label=f"{client}")#,marker='^', markevery=1, linewidth=2)
    
    axs[0].set_title("Send")
    axs[1].set_title("Receive")
    axs[0].set_ylabel("MB")
    axs[1].set_ylabel("MB")
    axs[1].set_xlabel("Round")
    legend = fig.legend(clients, 
                        loc="upper center", 
                        bbox_to_anchor=(0.5, 1.0), 
                        ncol=len(clients),
                        fontsize=12,)
    
    fig_server, axs_server = plt.subplots(2, 1, figsize=(20, 5))
    server_sends, server_receives = [], []
    for r in rounds:
        send_sum = server_filter[server_filter['round'] == r].send.sum()/1024
        recv_sum = server_filter[server_filter['round'] == r].receive.sum()/1024
        server_sends.append(send_sum)
        server_receives.append(recv_sum)
    axs_server[0].bar(rounds, server_sends, label="Server")#,marker='^', markevery=1, linewidth=2)
    axs_server[1].bar(rounds, server_receives, label="Server")#,marker='^', markevery=1, linewidth=2)
    axs_server[0].set_title("Send")
    axs_server[1].set_title("Receive")
    axs_server[0].set_ylabel("MB")
    axs_server[1].set_ylabel("MB")
    axs_server[0].set_xlabel("Round")
    legend = fig_server.legend(["Server"], 
                        loc="upper center", 
                        bbox_to_anchor=(0.5, 1.0), 
                        ncol=1,
                        fontsize=12,)
    
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
    
def plot_send_receive_evolution_round(time_df):
    rounds = time_df["round"].unique()
    sends, receives = [], []
    for round in rounds:
        df = time_df[time_df["round"] == round]
        sends.append(df["send"].sum()/1024)
        receives.append(df["receive"].sum()/1024)
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].plot(rounds, sends, label="Send", marker='^', markevery=1, linewidth=2)
    axs[1].plot(rounds, receives, label="Receive",marker='^', markevery=1, linewidth=2)
    axs[0].set_title("Send Evolution")
    axs[1].set_title("Receive Evolution")
    axs[0].set_ylabel("MB")
    axs[1].set_ylabel("MB")
    plt.show()
    
def plot_send_receive_round(df, round_number=1):
    info_round = df[df["round"] == round_number]
    send = info_round["send"]/1024
    receive = info_round["receive"]/1024
    time = info_round["timestamp"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].bar(time,send, label="Send in KB")
    axs[1].plot(time,receive, label="Receive in KB")
    axs[0].set_title(f"Send in round {round_number}")
    axs[1].set_title(f"Receive in round {round_number}")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    axs[0].set_ylabel("MB/s")
    axs[1].set_ylabel("MB/s")
    plt.show()