import os
import json
from omegaconf import DictConfig, OmegaConf
import torch
import torchvision.transforms as transforms
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.distributions import Dirichlet
#from flwr.common import NDArray, Scalar, Config
from typing import Any, Dict, Tuple, List
from torchvision import datasets
from torch.utils import data
from flwr.server.strategy import FedAdam, FedAvg, FedProx
from datasets import load_from_disk
import sys
sys.path.append("/Users/mathildepro/Documents/code_projects/ai-energy-consumption-framework/Jetson/utils")
from utils.dataset import MyDataset


class DataSetHandler:
    def __init__(self, config):
        self.dataname = config["data_name"]
        self.data_dir = config["download_dir"]
        self.num_clients = config["num_clients"]
        self.validation_split = config["validation_split"]
        self.batch_size = config["batch_size"]
        self.alpha = config["alpha"]
        self.partition = config["partition"]
        self.dataloaders = config["dataloaders"]
        self.partition_dir = config["partition_dir"]
        self.download_dir = config["download_dir"]
        #self.output_dir = config["output_dir"]
        
        self.idx_map = {}
        
    def __call__(self):
        return self.get_data()
    
    def __str__(self) -> str:
        return "Dataset: {} | Data_Dir: {} | Num Clients: {} | Validation Split: {}".format(self.dataname, self.data_dir, self.num_clients, self.validation_split)
    
    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)
        
    def get_data(self):
        
        ds = load_from_disk(self.download_dir)
        logging.info("dataset loaded")
        trainset = MyDataset(ds, 'train', 256)
        testset = MyDataset(ds, 'validation', 256)
        logging.info("NUM CLASSES : {}".format(len(np.unique(trainset.targets))))
        trainchunks, testdata = self._get_data(trainset, testset, self.partition, dataloaders=self.dataloaders)
        self.plot_label_distribution(trainset)
        with open(os.path.join(self.partition_dir,"idx_map.json"), "w") as f:
            json.dump(self.idx_map, f)
        return trainchunks, testdata
    
    def _get_data(self, traindata, testdata, partition_type, dataloaders = False):
        """_summary_

        Args:
            traindata (torchdataset): train data
            testdata (torchdataset): test
            partition_type (str: _description_

        Returns:
            Tuple[List[data.DataLoader], List[data.DataLoader], data.DataLoader] 
        """
        if partition_type == "iid":
            datasets = self.iid_partition(traindata)
        logging.info("DATA LOADED AND {} PARTITIONED".format(partition_type.upper()))
        return datasets, testdata
    
    def iid_partition(self, traindata) -> List[data.Subset]:
        partition_size = len(traindata) // self.num_clients
        lengths = [partition_size] * self.num_clients
        client_chunks = data.random_split(traindata, lengths=lengths, generator=torch.Generator().manual_seed(2024))
        for i in range(self.num_clients):
            self.idx_map.update({f"client_{i}": client_chunks[i].indices})
            #logging.info("IID Partitioning :{} DataSamples Per Clients".format(len(client_chunks[i])))
        logging.info("IID Partitioning :{} DataSamples Per Clients".format(partition_size))
        return client_chunks
    
    def _data_partition(self, traindata, testdata) -> Tuple[List[data.DataLoader], List[data.DataLoader], data.DataLoader]:
        partition_size = len(traindata) // self.num_clients
        lengths = [partition_size] * self.num_clients
        self.datasets = []
        for i in range(self.num_clients):
            datasetidx = [i * partition_size + t for t in range(partition_size)]
            self.datasets.append(data.Subset(traindata, datasetidx))
        logging.info("{} DataSamples Per Clients".format(partition_size))
        return self.datasets

    
    
def load_clientdata_from_file(config:DictConfig, client_id:int)->Tuple[data.DataLoader, data.DataLoader]:
    path_to_data = config.partition_dir
    validation_split = config.validation_split
    path_client = os.path.join(path_to_data, f"client_{client_id}")
    traindata = torch.load(os.path.join(path_client,"trainset_" + str(client_id) + ".pt"))
    # SPLIT TO VALIDATION
    len_val = len(traindata)//validation_split 
    len_train = len(traindata) - len_val
    lengths = [len_train, len_val]
    train, val = data.random_split(traindata, lengths, torch.Generator().manual_seed(2024))
    #logging.info("CLIENT {} TRAIN_SAMPLES: {} VALIDATION_SAMPLES: {}".format(client_id, len(train), len(val)))
    trainloader = data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
    valloader = data.DataLoader(val, batch_size=config.batch_size, shuffle=True)
    return trainloader, valloader 
        
def load_testdata_from_file(config:DictConfig):
    path_to_data = config.partition_dir
    testdata = torch.load(os.path.join(path_to_data,"testset.pt"))
    logging.info("TOTAL TEST SAMPLES : {}".format(len(testdata)))
    testloader = data.DataLoader(testdata, batch_size=config.batch_size, shuffle=True)
    return testloader

