from json import load
import torch
from torch.utils.data import DataLoader, random_split
import random
from torch.utils import data
from torchvision import datasets, transforms
from typing import List
from torchvision.datasets import CIFAR10
from torch.distributions import Dirichlet
import logging
import numpy as np

def label_skew_dirichlet(traindata, alpha:float, num_clients) -> List[data.Subset]:                
    nb_samples = len(traindata)
    nb_classes = len(np.unique(traindata.targets))
    print("Number of Classes : ", nb_classes)
    targets = np.array(traindata.targets)
    partition_size = nb_samples // num_clients
    min_size = 0
    min_require_sample = int(partition_size*0.1)
    it_count = 0
    print("Alpha : ", alpha)
    while min_size < min_require_sample:
        # Dirichlet distribution
        idx_batch = [[] for _ in range(num_clients)]
        lb_distribution = Dirichlet(torch.ones(num_clients) * alpha).sample(sample_shape=(nb_classes,)) # type: ignore
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
    for i in range(num_clients):
        random.shuffle(idx_batch[i])
        #self.idx_map.update({f"client_{i}": idx_batch[i]})
    client_chunks = [data.Subset(traindata, idx_batch[cid]) for cid in range(num_clients)]
    logging.info("Non-IID Partitioning : Label Skew, Min DataSamples Per Clients {}, Min Require Sample {}".format(min_size, min_require_sample))
    return client_chunks


def load_dataset(data_path:str, NUM_CLIENTS:int, BATCH_SIZE:int, iid:bool=True):
    train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                    transforms.RandomHorizontalFlip(),
    ])
    test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
    if iid:
        partition_size = len(trainset) // NUM_CLIENTS
        lengths = [partition_size] * NUM_CLIENTS
        dataset = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    else:
        dataset = label_skew_dirichlet(trainset, 0.5, NUM_CLIENTS)
    trainloaders = []
    valloaders = []
    for it,ds in enumerate(dataset):
        len_val = len(ds)//10 # validation 10%
        len_train = len(ds) - len_val
        train, val = random_split(ds, [len_train, len_val], torch.Generator().manual_seed(42))
        trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        valloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
        trainloaders.append(trainloader)
        valloaders.append(valloader)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    print("Data loaded and partitioned")
    return trainloaders, valloaders, testloader

# a,b,c = load_dataset("../data", 100, 12)
# print(len(a[0]), len(b[0]), len(c))
# ts = next(iter(a[0]))
# print(ts[0].shape)
