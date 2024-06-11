# from filter_exp import read_summaryfile, create_json_file
# from result_utils import (
#     read_server, 
#     read_data_from_dict,
#     compute_strategy_averages_results, 
#     plot_results,
#     filter_time_accuracy_all
# )
# from comm_utils import (
#     load_client_data,
#     load_server_data,
#     network_log_to_csv,
#     client_log_file_to_pd,
#     server_log_file_to_csv,
# )
from box import Box
import pandas as pd
import os
import sys
import cProfile
sys.path.append("/Users/Slaton/Documents/grenoble-code/fl-flower/jetson-tl/analysis/")


#client_list = ['client_0', 'client_1', 'client_2', 'client_3', 'client_4', 'client_5', 'client_6', 'client_7', 'client_8', 'client_9']
usr_homedir = "/Users/Slaton/Documents/grenoble-code/fl-flower"
parent_path = "/Users/Slaton/Documents/grenoble-code/fl-flower/energyfl/outputcifar10/10clients/"

match_condition = lambda summary:(
    (
        ((summary['strategy'].isin(['fedavg','fedyogi','fedadagrad'])) & 
        (summary['client.lr'] == 0.0316) & 
        (((summary['client.local_epochs'] == 1) & (summary['params.num_rounds'] == 300)) | 
        ((summary['client.local_epochs'].isin([5])) & (summary['params.num_rounds'] == 100))))
        | 
        ((summary['strategy'] == 'fedadam') & (summary['client.lr'] == 0.0316) & 
        (((summary['client.local_epochs']==5) & (summary['params.num_rounds'] == 100)) | (summary['client.local_epochs']==1)))
        |
        (((summary['strategy'] == 'fedconstraints') & (summary['client.lr'] == 0.0316) & (summary['client.local_epochs'].isin([1,5]))))
        | 
        (((summary['strategy'] == 'fedavg_adam') & (summary['client.lr'] == 0.001) & (summary['client.local_epochs'].isin([1,5]))))
    )
)

strategies = ['fedavg','fedyogi','fedadagrad', 'fedadam', 'fedconstraints','fedavg_adam']

#strategies = ['fedadam','fedconstraints']

from utils.process_experiment import ProcessResults, EnergyResults
from utils.process_data_functions import (
    read_path_from_dict, 
    read_server,
    process_rounds_time,
    process_rounds_time_from_log,
    process_host_round_time,
)
from utils.read_file_functions import load_client_data, load_server_data, server_log_file_to_csv
from utils.process_energy import (
    merge_exp_perf, 
    merge_client_training_perf, 
    concat_client_training_perf,
    compute_exp_energy_per_host, 
    compute_strategy_exp_energy_perf, 
    compute_energy_within_range,
    compute_exp_energy_perf,
    aggregate_round_stats
)
results_summary = EnergyResults(output_dir=parent_path, 
                                 usr_homedir=usr_homedir,
                                 strategies=strategies,
                                 condition=match_condition,
                                 threshold=0.75,
                                 epoch_list=[1,5],
                                 split='labelskew')
strategies_dict = results_summary.strategies_dict
#perf_summ = merge_exp_perf(results_summary, 'fedyogi')
#strategies = results_summary.strategies_dict.keys()
#host_summary, perf_summary = compute_exp_energy_perf(results_summary, 'fedyogi', 'epoch_5', 'exp_0')
results, hosts_summary, round_results = aggregate_round_stats(results_summary)
#config = aggregate_round_stats(results_summary)
#perf_summ = merge_exp_perf(results_summary, strategy)
#config = perf_summ.iloc[0]

# server_data = load_server_data(strategies_dict, 'fedadam', 'epoch_1', 'exp_0', filter=True)
# server_round_time = process_rounds_time(server_data['rounds_time'],mode='round')
# server_data['rounds_time']#.iloc[1124]
# server_round_time

# server_logs = server_data['logs']
# ok = process_rounds_time_from_log(server_logs)
# ok
# strategies_dict['fedadam']['split_epoch']['epoch_1']['exp_0']

