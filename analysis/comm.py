from comm_utils import load_server_data, load_client_data, process_rounds_time, filter_round_time, process_host_round_time, read_server_clients_data, plot_multples_clients, plot_send_receive_evolution_round
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from types import SimpleNamespace
from box import Box
import json
import sys
from filter_exp import read_summaryfile
from result_utils import read_server
import seaborn as sns
sns.set_theme(style="whitegrid")

client_list = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5', 'client_6', 'client_7', 'client_8', 'client_9']


parent_path = "/Users/Slaton/Documents/grenoble-code/fl-flower/energyfl/outputcifar10/10clients/comm/fedsfw/labelskew"
user = '/Users/Slaton/Documents/grenoble-code/fl-flower/'

summary = read_summaryfile(parent_path,usr_homedir=user, condition=None)
first_exp = summary.iloc[0]
exp_path = first_exp['result_folder']#.replace("/Users/Slaton/", user)
files = read_server_clients_data(exp_path)

exp_duration = first_exp['timestamps.end_experiment'] - first_exp['timestamps.start_experiment']

server_round_time = process_rounds_time(files.server.time)

def process_network_data(client_list, files):
    hosts_send = {}
    hosts_receive = {}
    hosts_round_time = {}
    for client in client_list:
        host_stat = process_host_round_time(files_holder=files, host=client)
        hosts_send[client] = host_stat['send']
        hosts_receive[client] = host_stat['receive']
        hosts_round_time[client] = host_stat['round time']
    # Client 
    send_df = pd.DataFrame(hosts_send)
    receive_df = pd.DataFrame(hosts_receive)
    return send_df, receive_df


def melt_send_receive(send_df, receive_df):
    mydf = []
    for (stat,df) in zip(['send','receive'],[send_df, receive_df]):
        df['round'] = df.index
        df_melt = df.melt(id_vars='round',var_name='client',value_name='value')
        df_melt['status'] = stat
        mydf.append(df_melt)
    concat_df = pd.concat([mydf[0], mydf[1]])
    concat_df['client'] = concat_df['client'].str.replace('client_','Client ')
    return concat_df

send_df, receive_df = process_data(client_list, files)
df_send_receive = avg_send_receive(send_df, receive_df)
#plot 
plt.figure(figsize=(10,5))
sns.boxplot(x='client', hue='status',y='value', data=df_send_receive)
plt.xlabel('Client')
plt.ylabel('Total Bandwidth/Round (MB)')
plt.legend(loc='upper center',ncol=2,bbox_to_anchor=(0.5, 1.1))

#Plot Barplot total send and receive per client
def sum_send_receive(send_df, receive_df):
    mydf = []
    for (stat,df) in zip(['send','receive'],[send_df, receive_df]):
        sum_status = df.sum(axis=0)
        sum_status = sum_status.drop('round')
        sum_status = sum_status.reset_index()
        sum_status.columns = ['client', 'value']
        sum_status['status'] = stat
        mydf.append(sum_status)
    sum_status = pd.concat([mydf[0], mydf[1]])
    sum_status['client'] = sum_status['client'].str.replace('client_','Client ')
    return sum_status

sum_send_total = sum_send_receive(send_df, receive_df)
plt.figure(figsize=(10,5))
sns.barplot(x='client', y='value', hue='status', data=sum_send_total)
plt.xlabel('Client')
plt.ylabel('Total Bandwidth (MB)')
plt.legend(loc='upper center',ncol=2,bbox_to_anchor=(0.5, 1.1))



#server
server_send_df, server_receive_df = process_data(['server'], files)
df_server_send_receive = avg_send_receive(server_send_df, server_receive_df)
#plot 
plt.figure(figsize=(10,5))
sns.boxplot(hue='status',y='value', data=df_server_send_receive)
plt.xlabel('Server')
plt.ylabel('Total Bandwidth/Round (MB)')
plt.legend(loc='upper center',ncol=2,bbox_to_anchor=(0.5, 1.1))


#Plot result
results_path = os.path.join(exp_path, 'server/results.pkl')
result_df = read_server(results_path)

fig, axs = plt.subplots(2, figsize=(10,10))
# Plot accuracy
axs[0].plot(result_df['server_round'], result_df['acc_centralized'], label='Centralized Accuracy')
axs[0].plot(result_df['server_round'], result_df['acc_distributed'], label='Distributed Accuracy')
axs[0].set_ylabel('Accuracy')
# Plot loss
axs[1].plot(result_df['server_round'], result_df['losses_centralized'], label='Centralized Loss')
axs[1].plot(result_df['server_round'], result_df['losses_distributed'], label='Distributed Loss')
axs[1].set_ylabel('Loss')
legend = fig.legend(loc='upper center',ncol=2,
                    labels=['Centralized', 'Distributed'],
                    bbox_to_anchor=(0.5, 1.))
plt.tight_layout()
plt.show()

