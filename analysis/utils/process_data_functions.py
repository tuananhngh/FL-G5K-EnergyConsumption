import pandas as pd
from box import Box
from .read_file_functions import (
    read_server_clients_data,
    flwr_pkl
)

def process_host_round_time(files_holder, host:str):
    """
    This function processes the round time for a specific host in the experiment.
    Args:
        files_holder (SimpleNamespace): The SimpleNamespace object containing the server and client data.
        host (str): The ID of the host for which to process the round time. (e.g., 'client_1')
    """
    server_round_time = process_rounds_time(files_holder.server.time, mode='round')
    host_round_time = filter_round_time(server_round_time, files_holder[host].network)
    send, receives = [], []
    rounds = host_round_time['round'].unique()
    rounds_time = host_round_time['round time'].unique()
    for r in rounds:
        send_r = host_round_time[host_round_time['round'] == r]['send'].sum()/1024
        receive_r = host_round_time[host_round_time['round'] == r]['receive'].sum()/1024
        send.append(send_r)
        receives.append(receive_r)
    host_statistics = pd.DataFrame({'round': rounds, 
                                    'send': send, 
                                    'receive': receives, 
                                    'round time': rounds_time})
    return host_statistics

def process_rounds_time_from_log(log:pd.DataFrame):
    """
    Take the server logs dataframe and return the time taken for each round
    Args:
        logs (pd.DataFrame): The server logs dataframe.
    Returns:
        pd.DataFrame: The dataframe containing the time taken for each round.
        Columns: ['round', 'fittime', 'evaltime','roundtime','starttime','endtime']
    """
    n_rounds = log.round_number.unique()
    server_round_time = pd.DataFrame()
    for r in n_rounds:
        round_info = log[log.round_number == r]
        roundstart = round_info.timestamp.min()
        roundend = round_info.timestamp.max()
        roundtime = (roundend - roundstart).total_seconds()
        rounddict = {'round': r, 'roundtime': roundtime, 'starttime': roundstart, 'endtime': roundend}
        for status in ['fit', 'evaluate']:
            round_mode_info = round_info[round_info.round_mode == status]
            start = round_mode_info.timestamp.min()
            end = round_mode_info.timestamp.max()
            time = (end-start).total_seconds()
            timekey = f"{status}time"
            timestart = f"{status}start"
            timeend = f"{status}end"
            rounddict[timekey] = time
            rounddict[timestart] = start
            rounddict[timeend] = end
        server_round_time = pd.concat([server_round_time, pd.DataFrame(rounddict, index=[0])], ignore_index=True)
    return server_round_time
    

def process_rounds_time(rounds_time:pd.DataFrame, mode='round'):
    """
    Take the server rounds_time dataframe and return time taken for each round
    mode : ['round', 'fit', 'eval_dis', 'eval_cen', 'total']
    round : take total duration of each round from fit call to distributed_evaluated_aggregated
    fit : take duration of each round from fit call to fit aggregated
    eval_dis : take duration of each round from distributed evaluate call to distributed evaluated aggregated
    eval_cen : take duration of each round from central evaluate call to central evaluate end
    """
    rounds_time_pivot = rounds_time.pivot(index="round", columns="status", values="timestamp").reset_index()
    rounds_time_pivot.columns = rounds_time_pivot.columns.str.replace(' ','_')
    rounds_time_pivot = rounds_time_pivot.rename(columns={"start_fit_call": "fit_call", 
                                                          "res_fit_aggregated": "fit_agg", 
                                                          "res_fit_received": "fit_received", 
                                                          "central_evaluate_call": "eval_cen_call", 
                                                          "central_evaluated": "eval_cen_received",
                                                          "central_evaluate_end": "eval_cen_agg",
                                                          "distributed_evaluate_call": "eval_dis_call", 
                                                          "distributed_evaluated": "eval_dis_received",
                                                          "distributed_evaluate_end": "eval_dis_agg",})
    status = ["fit","eval_dis","eval_cen"]
    for m in status:
        rounds_time_pivot[f"{m}_duration"] = (rounds_time_pivot[f"{m}_agg"] - rounds_time_pivot[f"{m}_call"]).dt.total_seconds()
        rounds_time_pivot[f"{m}_call_received"] = (rounds_time_pivot[f"{m}_received"] - rounds_time_pivot[f"{m}_call"]).dt.total_seconds()
        rounds_time_pivot[f"{m}_received_agg"] = (rounds_time_pivot[f"{m}_agg"] - rounds_time_pivot[f"{m}_received"]).dt.total_seconds()
    rounds_time_pivot["round_duration"] = (rounds_time_pivot.eval_dis_agg - rounds_time_pivot.fit_call).dt.total_seconds()
    
    if mode == 'round':
        df = rounds_time_pivot[['round', 'fit_call', 'eval_dis_agg','round_duration']]
        df = df.rename(columns={"round":"Server Round","fit_call": "Start Time", "eval_dis_agg": "End Time"})
        return df
    elif mode == 'total':
        df = rounds_time_pivot
        return df
    else:
        df = rounds_time_pivot[['round', f'{mode}_call', f'{mode}_agg',f'{mode}_duration']]
        df = df.rename(columns={"round":"Server Round",f"{mode}_call": "Start Time", f"{mode}_agg": "End Time"})
    return df

def filter_round_time(roundtime_df, tofilter_df):
    """
    Filter the dataframe tofilter_df based on the start and end time in roundtime_df
    """
    tofilter_df["timestamp"] = pd.to_datetime(tofilter_df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
    dfs = []
    for idx, row in roundtime_df.iterrows():
        start_time = row["Start Time"]
        end_time = row["End Time"]
        df = tofilter_df[(tofilter_df["timestamp"] >= start_time) & (tofilter_df["timestamp"] <= end_time)].copy()
        df.loc[:, 'round'] = row["Server Round"]
        df.loc [:, 'round time'] = (end_time - start_time).total_seconds()
        dfs.append(df)
    final_df = pd.concat(dfs)
    return final_df


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

def melt_send_receive_df(send_df, receive_df):
    mydf = []
    for (stat,df) in zip(['send','receive'],[send_df, receive_df]):
        df['round'] = df.index
        df_melt = df.melt(id_vars='round',var_name='client',value_name='value')
        df_melt['status'] = stat
        mydf.append(df_melt)
    concat_df = pd.concat([mydf[0], mydf[1]])
    concat_df['client'] = concat_df['client'].str.replace('client_','Client ')
    return concat_df

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

def get_bandwidth(strategy:str, epoch:str, strategies_dict:Box):
    summary_info = strategies_dict.__getattr__(strategy).split_epoch.__getattr__(epoch).exp_0.summary
    exp_path = summary_info.result_folder
    exp_keys = strategies_dict.__getattr__(strategy).split_epoch.__getattr__(epoch).keys()
    exp_duration = summary_info.timestamps_end_experiment - summary_info.timestamps_start_experiment
    files = read_server_clients_data(exp_path) #change this
    server_round_time = process_rounds_time(files.server.time)
    
    client_list = [key for key in exp_keys if 'client' in key]
    send_df, receive_df = process_network_data(client_list, files)
    df_send_receive = melt_send_receive_df(send_df, receive_df)
    df_send_receive_sum = sum_send_receive(send_df, receive_df)
    return df_send_receive, df_send_receive_sum, server_round_time, exp_duration


def get_bandwidth_server(strategy:str, epoch:str, strategies_dict:Box):
    summary_info = strategies_dict.__getattr__(strategy).split_epoch.__getattr__(epoch).exp_0.summary
    exp_path = summary_info.result_folder
    exp_duration = summary_info.timestamps_end_experiment - summary_info.timestamps_start_experiment
    files = read_server_clients_data(exp_path) #change this
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


def read_path_from_dict(strat_dict, strategy, epoch, exp, host, file_type):
    """
    Read data from a dictionary containing paths to data files.
    """
    path = strat_dict[strategy]['split_epoch'][epoch][exp][host][file_type]
    return path

def concat_server_results(strategy, epoch, host, file_type, strategies_dict, n_exp = 5):
    """
    Concatenate server results from multiple experiments.
    """
    result_paths = [read_path_from_dict(strategies_dict, strategy, epoch, f'exp_{i}' , host, file_type) for i in range(n_exp)]
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
