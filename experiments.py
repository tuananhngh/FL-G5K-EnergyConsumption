"""Module for running experiments on Grid'5000"""
from os import path
from execo import SshProcess
import sys

# Start the process
# process = SshProcess("python3 /home/mjay/FL-G5K-Test/xav_read_power.py", host=server, connection_params={'user':'root'})
# process.start()

# To get the data (later)
# import requests
# request_start = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time() - 1000))
# request_stop = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time() -5))
# node = "estats-5"

# url = f"https://api.grid5000.fr/stable/sites/toulouse/metrics?nodes={node}&start_time={request_start}&end_time={request_stop}"
# # url = "https://api.grid5000.fr/stable/sites/toulouse/metrics/?"
# r = requests.get(url, verify=False).json()
 
def concat_dict(dict_list):
    new_dict = {}
    for d in dict_list:
        if isinstance(d, (dict)):
            new_dict.update(d)
    return new_dict

def execute_command_on_server_and_clients(hosts, command, log_file="/tmp/logs.log", background=False):
    # check if log file exists
    if not path.exists(log_file):
        # create file
        with open(log_file, 'x') as f:
            f.write("")
    processes = []
    for host in hosts:
        process = SshProcess(
            command, 
            host=host, 
            connection_params={'user':'root'},            
            stdout_handlers=[sys.stdout, log_file], 
            stderr_handlers=[sys.stderr, log_file]
            )
        processes.append(process)
        if background:
            process.start()
        else:
            process.run()
            if process.ok:
                print(f"Successfully executed on {host} command '{command}'")
            else:
                process.kill()
                print(f"Failed to execute on {host} command '{command}'")
                print(process.stderr)
                print(process.stdout)
    return processes