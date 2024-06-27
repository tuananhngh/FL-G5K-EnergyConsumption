import os
from omegaconf import DictConfig
import torch
import logging
import sys
#from flwr.common import NDArray, Scalar, Config
from typing import Any, Dict, Tuple, List
from torch.utils import data
#from utils.dataset import MyDataset
import torchvision.transforms as transforms
from datasets import load_from_disk

#os.chdir('/home/tunguyen/jetson-imagenet/src/utils')

class MyDataset(torch.utils.data.TensorDataset):
    def __init__(self, dataset, split="train", resolution=256):
        # Loads the dataset that needs to be transformed
        self.dataset = dataset[split]
        self.resolution = resolution
        self.targets = self.dataset['label']
        

    def __getitem__(self, idx):
        logging.debug("getting item")
        # Sample row idx from the loaded dataset
        try :
            sample = self.dataset[idx]
        except OSError as e:
            logging.error(f"Error while loading image: {e}")
            logging.error(f"Index: {idx}")
            logging.error("Repeating previous index")
            sample = self.dataset[idx-1]
        # Split up the sample example into an image and label variable
        data, label = sample['image'], sample['label']
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),  # Resize to size 256x256
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),  # Convert all images to RGB format
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Transform image to Tensor object
            normalize,
        ])
        
        
        # Returns the transformed images and labels
        return transform(data), torch.tensor(label)

    def __len__(self):
        return len(self.dataset)

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

#ok = load_testdata_from_file(DictConfig({'partition_dir': '/home/tunguyen/jetson-imagenet/data', 'batch_size': 32}))