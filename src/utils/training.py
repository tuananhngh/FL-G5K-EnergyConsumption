import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import random
from tqdm import tqdm
from flwr.common import NDArray, Scalar, Config
from typing import Any, Dict, Tuple, List
from collections import OrderedDict
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
    state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
    #state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})
    net.load_state_dict(state_dict, strict = True)
    
def get_parameters(net) -> List[NDArray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
