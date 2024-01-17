import os
from matplotlib.ft2font import LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.distributions import Dirichlet
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR, ExponentialLR
from tqdm import tqdm
from flwr.common import NDArray, Scalar, Config
from typing import Any, Dict, Tuple, List
from collections import OrderedDict
from torchvision import datasets
from torch.utils import data
from flwr.server.strategy import FedAdam, FedAvg, FedProx


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Creating seeds to make results reproducible
def seed_everything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# seed = 2024
# seed_everything(seed)

# with open("config/config_file.yaml", "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
#     ok = OmegaConf.create(config)

class DataSetHandler:
    def __init__(self, config:Config):
        self.dataname = config["data_name"]
        self.data_dir = config["data_dir"]
        self.num_clients = config["num_clients"]
        self.validation_split = config["validation_split"]
        self.batch_size = config["batch_size"]
        self.alpha = config["alpha"]
        self.partition = config["partition"]
        
        self.idx_map = {}
        
    def __call__(self):
        return self.get_dataloader()
    
    def __str__(self) -> str:
        return "Dataset: {} | Data_Dir: {} | Num Clients: {} | Validation Split: {}".format(self.dataname, self.data_dir, self.num_clients, self.validation_split)
    
    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)
    
    
    def iid_partition(self, traindata) -> List[data.Subset]:
        partition_size = len(traindata) // self.num_clients
        lengths = [partition_size] * self.num_clients
        client_chunks = data.random_split(traindata, lengths=lengths)
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
        min_require_sample = int(partition_size*0.1)
        print("Alpha : {} Min require sample : {}".format(alpha, min_require_sample))
        it_count = 0
        while min_size < min_require_sample:
            # Dirichlet distribution
            lb_distribution = Dirichlet(torch.ones(self.num_clients) * alpha).sample()
            lb_distribution = lb_distribution/lb_distribution.sum()
            print("Sample Distribution : ", (lb_distribution*nb_samples).int())
            min_size = min((lb_distribution*nb_samples).int())
            it_count += 1
            if it_count % 1000 == 0:
                print("Iteration: {}, Min Size: {}, Min Require Sample {}".format(it_count, min_size, min_require_sample))
        proportions = (torch.cumsum(lb_distribution,dim=0) * nb_samples).int()
        print("Proportions : ", proportions)
        splits = np.split(np.arange(nb_samples), proportions)
        for i in range(self.num_clients):
            self.idx_map.update({f"client_{i}": splits[i]})
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

    def _get_dataloader(self, traindata, testdata, partition_type):
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
        trainloaders = []
        valloaders = []
        for (it,ds) in enumerate(datasets):
            len_val = len(ds) // self.validation_split # 10% of the dataset
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            train, val = data.random_split(ds, lengths)
            
            trainloader = data.DataLoader(train, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker)
            valloader = data.DataLoader(val, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker)
            trainloaders.append(trainloader)   
            valloaders.append(valloader)
        testloader = data.DataLoader(testdata, batch_size=self.batch_size, shuffle=True) 
        logging.info("DATA LOADED AND {} PARTITIONED".format(partition_type.upper()))
        return trainloaders, valloaders, testloader
    
    def cifar10_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
        output_dir = os.getcwd()
        print("Image Saving at : ", output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/distribution_per_client.png")
        
    def get_dataloader(self):
        transform = self.__getattribute__(self.dataname.lower() + "_transform")()
        trainset = datasets.__getattribute__(self.dataname)(root=self.data_dir, train=True, download=True, transform=transform)
        testset = datasets.__getattribute__(self.dataname)(root=self.data_dir, train=False, download=True, transform=transform)
        
        trainloaders, valloaders, testloaders = self._get_dataloader(trainset, testset, self.partition)
        self.plot_label_distribution(trainset)
        return trainloaders, valloaders, testloaders
    
        



def load_dataset(config:Dict[str,Scalar])->Tuple[List[data.DataLoader], List[data.DataLoader], data.DataLoader]:
#def load_dataset(num_clients,batch_size,validation_split,root_path):
    num_clients = config["num_clients"]
    batch_size = config["batch_size"]
    validation_split = config["validation_split"]
    root_path = config["root_data"]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trainset = CIFAR10(root=root_path, train=True, download=True, transform=transform)
    testsset = CIFAR10(root=root_path, train=False, download=True, transform=transform)

    # print(len(trainset))
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    # print(lengths)
    # if sum(lengths) != len(trainset):
    #     print("Sum of partionned data is not equal to original data")
    datasets = []
    for i in range(num_clients):
        datasetidx = [i * partition_size + t for t in range(partition_size)]
        datasets.append(data.Subset(trainset, datasetidx))
    logging.info("{} DataSamples Per Clients".format(partition_size))
    trainloaders = []
    valloaders = []
    for (it,ds) in enumerate(datasets):
        len_val = len(ds)//validation_split # 10% of the dataset
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        train, val = data.random_split(ds, lengths, torch.Generator().manual_seed(seed))
        trainloader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valloader = data.DataLoader(val, batch_size=batch_size, shuffle=True)
        #torch.save(trainloader, os.path.join(partition_data_path,"trainloader_" + str(it) + ".pt"))
        #torch.save(valloader, os.path.join(partition_data_path,"valloader_" + str(it) + ".pt"))
        trainloaders.append(trainloader)   
        valloaders.append(valloader)
    testloader = data.DataLoader(testsset, batch_size=batch_size, shuffle=False) 
    #torch.save(testloader, os.path.join(partition_data_path,"testloader.pt"))
    print("Dataset loaded and partitioned")
    return trainloaders, valloaders, testloader


def check_device(neural_net):
    is_model_on_gpu = next(neural_net.parameters()).is_cuda
    if is_model_on_gpu:
        print("Model is on GPU")
    else:
        print("Model is on CPU")


def train(model, trainloader, valloader, epochs, optimizer, device):
    model.to(device)
    check_device(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_samples = len(trainloader.dataset)
    for _ in tqdm(range(epochs)):
        epoch_loss, correct_prediction = 0, 0
        for dt,lb in trainloader:
            dt, lb = dt.to(device), lb.to(device)
            optimizer.zero_grad()
            outputs = model(dt)
            losses = criterion(outputs, lb)
            losses.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += losses.item()
            #total_samples += lb.size(0)
            correct_prediction += (torch.max(outputs.data, 1)[1] == lb).sum().item()
        epoch_loss = epoch_loss/len(trainloader)
        epoch_acc = correct_prediction / total_samples
    # Validation metrics
    val_loss, val_acc = validation(model, valloader, device)
    results = {
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    return results

def validation(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    total_samples = len(dataloader.dataset)
    #logging.info("VALIDATION DEVICE : {}".format(device))
    with torch.no_grad():
        val_loss, correct_prediction = 0.,0.
        for dt,lb in dataloader:
            dt, lb = dt.to(device), lb.to(device)
            outputs = model(dt)
            losses = criterion(outputs, lb)
            val_loss += losses.item()
            correct_prediction += (torch.max(outputs.data, 1)[1] == lb).sum().item()
    avg_loss = val_loss/len(dataloader)
    accuracy = correct_prediction / total_samples
    return avg_loss, accuracy

def test(model, dataloader, device, steps=None, verbose=True):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    total_samples=len(dataloader.dataset)
    with torch.no_grad():
        val_loss, correct= 0., 0.
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            losses = criterion(outputs, labels)
            val_loss += losses.item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
        #logging.info("TOTAL TEST SAMPLES : {}".format(tt))
    avg_loss = val_loss/len(dataloader)
    accuracy = correct/total_samples
    if verbose:
        logging.info("Loss: {} | Accuracy: {}".format(avg_loss, accuracy))
    #model.to("cpu")
    return avg_loss, accuracy

def set_parameters(net, parameters:NDArray) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    #state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
    state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})
    net.load_state_dict(state_dict, strict = True)
    
def get_parameters(net) -> List[NDArray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def load_dataloader(client_id, path_to_data):
    trainloader, valloader = torch.load(os.path.join(path_to_data,"trainloader_" + str(client_id) + ".pt")), torch.load(os.path.join(path_to_data,"valloader_" + str(client_id) + ".pt"))
    testloader = torch.load(os.path.join(path_to_data,"testloader.pt"))
    return trainloader, valloader, testloader

