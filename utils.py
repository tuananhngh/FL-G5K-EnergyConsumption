from json import load
import os
import sched
import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging
import numpy as np
import random
from torch.optim.lr_scheduler import LinearLR, ExponentialLR
from tqdm import tqdm
from flwr.common import NDArray, Scalar
from typing import Dict, Tuple, List
from collections import OrderedDict
from torchvision.datasets import CIFAR10
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

seed = 2024
seed_everything(seed)

    

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

#trainl,vall,testl=load_dataset(12,64,10,"./")

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
