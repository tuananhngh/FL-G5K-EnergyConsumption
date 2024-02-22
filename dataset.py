from json import load
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10


def load_dataset(data_path:str, NUM_CLIENTS:int, BATCH_SIZE:int):
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    dataset = random_split(trainset, lengths, torch.Generator().manual_seed(42))
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
