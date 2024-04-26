import pandas as pd
import numpy as np
import csv
from pathlib import Path

parent_path = Path("/home/tunguyen/energyfl/outputcifar10/10clients/fedyogi/labelskew/2024-04-18_21-00-03/client_host_8/")
path_to_log = parent_path/"network.log"
network_csv = parent_path/"network_file.csv"

client_log = parent_path/"client.log"
client_log_csv = parent_path/"client_logs.csv"

def read_network(path_to_log, file_save):
    with open(path_to_log, 'r') as log_file, open(file_save, 'w', newline='') as csv_file:
        log_lines = log_file.readlines()
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Process', 'Send', 'Receive'])  # Write header

        for line in log_lines:
            if line.startswith('2024'): 
                try:
                    date, time, process, value1, value2 = line.split(maxsplit=4)
                    if process.startswith('python3'):
                        timestamp = f"{date} {time}"
                        csv_writer.writerow([timestamp, process, value1, value2])
                except :
                    pass # skip the line if it does not have the correct format
            elif line.startswith('python3'):
                try:
                    process, value1, value2 = line.split(maxsplit=2)
                    timestamp = 'None'
                    csv_writer.writerow([timestamp, process, value1, value2])
                except :
                    pass

read_network(path_to_log, network_csv)

def client_log_to_csv(path_to_log, file_save):
    with open(path_to_log, 'r') as log_file, open(file_save, 'w', newline='') as csv_file:
        log_lines = log_file.readlines()        
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Message'])
        
        for line in log_lines:
            if '[flwr][INFO]' in line:
                try: 
                    timestamp, message = line.split(' - ', maxsplit=1)
                    new_timestamp = timestamp.replace('[flwr][INFO]', '')
                    new_timestamp = new_timestamp.strip('[]')
                    csv_writer.writerow([new_timestamp, message])
                except:
                    pass
    return pd.read_csv(file_save)
ok = client_log_to_csv(client_log, client_log_csv)


def read_network_csv(file_save):
    df = pd.read_csv(file_save)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Send'] = df['Send'].astype(float)
    df['Receive'] = df['Receive'].astype(float)
    return df

def read_client_log_csv(file_save):
    df = pd.read_csv(file_save)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

df = read_network_csv(file_save=network_csv)
client_df = read_client_log_csv(file_save=client_log_csv)