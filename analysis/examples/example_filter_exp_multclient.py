from box import Box
import pandas as pd
import os
import sys
from utils.process_experiment import ProcessResults, EnergyResults, read_server
from utils.process_energy import (
    aggregate_round_stats,
    compute_energy_within_range,
    compute_strategy_exp_energy_perf,
    compute_host_energy
)
from utils.read_file_functions import load_client_data, load_server_data, server_log_file_to_csv

sys.path.append("/Users/Slaton/Documents/grenoble-code/fl-flower/jetson-tl/analysis/")
usr_homedir = "/Users/Slaton/Documents/grenoble-code/fl-flower"
strategies = ['fedadam', 'fedconstraints']
nb_clients = [10,20,30,50,70,100]

# example_path = os.path.join(parent_path, f'{10}clients', {strategies[0]}, 'labelskew')




def read_multiclient_host(usr_homedir,parent_path, strategy, threshold=None):
    result_summary = EnergyResults(output_dir=parent_path,
                                usr_homedir=usr_homedir,
                                strategies=strategies,
                                condition=None,
                                threshold=threshold,
                                epoch_list=[1],
                                split='labelskew'
                                )
    exp_summary = result_summary.get_experiment_summary(strategy,'epoch_1','exp_0')
    energy_summary,perf_summary = compute_strategy_exp_energy_perf(result_summary, strategy)
    return energy_summary, perf_summary, exp_summary

parent_path = "/Users/Slaton/Documents/grenoble-code/fl-flower/energyfl/outputcifar10/30clients/fractionfit/"
rs30 = EnergyResults(output_dir=parent_path,
                     usr_homedir=usr_homedir,
                     strategies=strategies,
                        condition=None,
                        threshold=0.75,
                        epoch_list=[1],
                        split='labelskew')
exp_summary = rs30.get_experiment_summary('fedadam','epoch_1','exp_0')
strat_dict = rs30.strategies_dict
server_res_path = strat_dict['fedadam']['split_epoch']['epoch_1']['exp_0']['server']['results']
hehe = read_server(server_res_path)
hehe
                    


output_path = "/Users/Slaton/Documents/grenoble-code/fl-flower/energyfl/outputcifar10/"
save_path = "../files/"
def read_all_multiclients_host(output_path, save_path, usr_home_dir, nb_clients_list, strategy):
    energy_summaries = pd.DataFrame()
    perf_summaries = pd.DataFrame()
    exp_summaries = pd.DataFrame()
    for nb in nb_clients_list:
        print(f'Number of clients: {nb}')
        parent_path = os.path.join(output_path, f'{nb}clients', 'fractionfit')
        energy_summary, perf_summary, exp_summary = read_multiclient_host(usr_home_dir, parent_path, strategy,
                                                                             threshold=0.75)
        energy_summary["nb_clients"] = [nb]*len(energy_summary)
        perf_summary["nb_clients"] = [nb]*len(perf_summary)
        
        exp_summary = pd.DataFrame(exp_summary, index=[0])
        exp_summary["nb_clients"] = [nb]*len(exp_summary)
        energy_summaries = pd.concat([energy_summaries, energy_summary], axis=0)
        perf_summaries = pd.concat([perf_summaries, perf_summary], axis=0)
        exp_summaries = pd.concat([exp_summaries, exp_summary], axis=0)
    return energy_summaries, perf_summaries, exp_summaries
        
energy_summary, perf_summary, exp_summary = read_all_multiclients_host(output_path, save_path, usr_homedir, nb_clients, 'fedconstraints')

perf_summary[['total_kWh',"nb_clients"]].plot(x="nb_clients", y="total_kWh", kind="bar")

# energy_summary.to_csv(os.path.join(save_path,f'{nb}_clients_energy_summary.csv'))
# perf_summary.to_csv(os.path.join(save_path,f'{nb}_clients_perf_summary.csv'))
# exp_summary.to_csv(os.path.join(save_path,f'{nb}_clients_exp_summary.csv'))


