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
#from flwr.server.strategy import FedAdam, FedAvg, FedProx


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
        #self.output_dir = config["output_dir"]
        
        self.idx_map = {}
        
    def __call__(self):
        return self.get_data()
    
    def __str__(self) -> str:
        return "Dataset: {} | Data_Dir: {} | Num Clients: {} | Validation Split: {}".format(self.dataname, self.data_dir, self.num_clients, self.validation_split)
    
    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)
    
    
    def iid_partition(self, traindata) -> List[data.Subset]:
        partition_size = len(traindata) // self.num_clients
        lengths = [partition_size] * self.num_clients
        client_chunks = data.random_split(traindata, lengths=lengths, generator=torch.Generator().manual_seed(2024))
        for i in range(self.num_clients):
            self.idx_map.update({f"client_{i}": client_chunks[i].indices})
            #logging.info("IID Partitioning :{} DataSamples Per Clients".format(len(client_chunks[i])))
        logging.info("IID Partitioning :{} DataSamples Per Clients".format(partition_size))
        return client_chunks
    
    def label_skew_dirichlet(self, traindata, alpha:float) -> List[data.Subset]:                
        nb_samples = len(traindata)
        nb_classes = len(np.unique(traindata.targets))
        print("Number of Classes : ", nb_classes)
        targets = np.array(traindata.targets)
        partition_size = nb_samples // self.num_clients
        min_size = 0
        min_require_sample = int(partition_size*0.1)
        it_count = 0
        print("Alpha : ", alpha)
        while min_size < min_require_sample:
            # Dirichlet distribution
            idx_batch = [[] for _ in range(self.num_clients)]
            lb_distribution = Dirichlet(torch.ones(self.num_clients) * alpha).sample(sample_shape=(nb_classes,))
            for k in range(nb_classes):
                class_idx = np.where(targets == k)[0]
                check_clientsamples = [len(client_idx) < partition_size for client_idx in idx_batch]
                check_clientsamples = np.array(check_clientsamples)
                #print("Class Check {} : {}".format(k, condition))
                lb_class_balance = lb_distribution[k] * check_clientsamples # zero out client already have enough samples
                lb_class_balance = lb_class_balance/lb_class_balance.sum()
                #print("Class {} : {} Len Class {}".format(k, lb_class_balance, len(class_idx)))
                class_proportions = (torch.cumsum(lb_class_balance,dim=0) * len(class_idx)).int()
                client_splits = np.split(class_idx, class_proportions.tolist())
                #print("Class {} : {}".format(k, [len(split) for split in client_splits]))
                for i, split in enumerate(client_splits[:-1]):
                    idx_batch[i].extend(split.tolist())
            min_size = min([len(client_idx) for client_idx in idx_batch])
            it_count += 1
            if it_count % 1000 == 0:
                print("Iteration: {}, Min Size: {}, Min Require Sample {}".format(it_count, min_size, min_require_sample))
        for i in range(self.num_clients):
            random.shuffle(idx_batch[i])
            self.idx_map.update({f"client_{i}": idx_batch[i]})
        client_chunks = [data.Subset(traindata, idx_batch[cid]) for cid in range(self.num_clients)]
        logging.info("Non-IID Partitioning : Label Skew, Min DataSamples Per Clients {}, Min Require Sample {}".format(min_size, min_require_sample))
        return client_chunks
    
    def sample_skew_dirichlet(self, traindata, alpha:float) -> List[data.Subset]:
        nb_samples = len(traindata)
        partition_size = nb_samples // self.num_clients
        min_size = 0
        min_require_sample = int(partition_size*0.05)
        print("Alpha : {} Min require sample : {}".format(alpha, min_require_sample))
        it_count = 0
        while min_size < min_require_sample:
            # Dirichlet distribution
            lb_distribution = Dirichlet(torch.ones(self.num_clients) * alpha).sample()
            lb_distribution = lb_distribution/lb_distribution.sum()
            #print("Sample Distribution : ", (lb_distribution*nb_samples).int())
            min_size = min((lb_distribution*nb_samples).int())
            it_count += 1
            if it_count % 1000 == 0:
                print("Iteration: {}, Min Size: {}, Min Require Sample {}".format(it_count, min_size, min_require_sample))
        proportions = (torch.cumsum(lb_distribution,dim=0) * nb_samples).int()
        #print("Proportions : ", proportions)
        splits = np.split(np.arange(nb_samples), proportions)
        for i, split in enumerate(splits[:-1]):
            #logging.info("Type : {} | Len : {}".format(type(splits[i]), len(splits[i])))
            self.idx_map.update({f"client_{i}": split.tolist()})
        client_chunks = [data.Subset(traindata, splits[i]) for i in range(self.num_clients)] 
        logging.info("Non-IID Partitioning : Data Partition Skew, Min {} DataSamples Per Clients".format(min_size))
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
        elif partition_type == "label_skew":
            datasets = self.label_skew_dirichlet(traindata, alpha=self.alpha)
        elif partition_type == "sample_skew":
            datasets = self.sample_skew_dirichlet(traindata, alpha=self.alpha)
        logging.info("DATA LOADED AND {} PARTITIONED".format(partition_type.upper()))
        return datasets, testdata
    
    def cifar10_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomHorizontalFlip(),
        ])
        return transform
    
    def cifar100_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomHorizontalFlip(),
        ])
        return transform
    
    def mnist_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform
    
    def fashionmnist_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform
    
    def tinyimagenet_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ])
        return transform

    def plot_label_distribution(self, traindata):
        # Initialize a list to hold the counts for each client
        client_counts = {}
        targets = np.array(traindata.targets)
        nb_classes = len(np.unique(traindata.targets))
        for key in self.idx_map.keys():
            # Get the indices for this client
            indices = self.idx_map[key]
            labels = targets[indices]
            counts = np.bincount(labels, minlength=nb_classes)  # Assuming 10 classes
            client_counts[key] = counts

        num_clients = len(client_counts)
        num_cols = 2
        num_rows = num_clients // num_cols + num_clients % num_cols
         
        # Plot the label distribution for each client
        plt.figure(figsize=(10, num_rows * 5))
        for i, (client, counts) in enumerate(client_counts.items()):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.bar(np.arange(nb_classes), counts)
            plt.title(f'{client} Label Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
        plt.tight_layout()
        #plt.show()
        output_dir = self.partition_dir
        print("Image Saving at : ", output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/distribution_per_client.png")
        
    def get_data(self):
        transform = self.__getattribute__(self.dataname.lower() + "_transform")()
        trainset = datasets.__getattribute__(self.dataname)(root=self.data_dir, train=True, download=True, transform=transform)
        testset = datasets.__getattribute__(self.dataname)(root=self.data_dir, train=False, download=True, transform=transform)
        logging.info("NUM CLASSES : {}".format(len(np.unique(trainset.targets))))
        trainchunks, testdata = self._get_data(trainset, testset, self.partition, dataloaders=self.dataloaders)
        self.plot_label_distribution(trainset)
        with open(os.path.join(self.partition_dir,"idx_map.json"), "w") as f:
            json.dump(self.idx_map, f)
        return trainchunks, testdata
    
    
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
    logging.info("CLIENT {} TRAIN_SAMPLES: {} VALIDATION_SAMPLES: {}".format(client_id, len(train), len(val)))
    trainloader = data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
    valloader = data.DataLoader(val, batch_size=config.batch_size, shuffle=True)
    return trainloader, valloader 
        
def load_testdata_from_file(config:DictConfig):
    path_to_data = config.partition_dir
    testdata = torch.load(os.path.join(path_to_data,"testset.pt"))
    logging.info("TOTAL TEST SAMPLES : {}".format(len(testdata)))
    testloader = data.DataLoader(testdata, batch_size=config.batch_size, shuffle=True)
    return testloader

