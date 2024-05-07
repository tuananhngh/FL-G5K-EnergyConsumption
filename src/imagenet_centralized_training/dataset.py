import logging
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

''' 
Creates a class object MyDataset
************
__init__
  This function will load a part of the ImageNet-1K dataset specified in the 
  class caller.
  Options are train, validation, and test.
************
__getitem__
  This function will return two variables containing each a variable present 
  in the original ImageNet-1K dataset.
  Before returning these variables, it will split up and transform the image 
  into a fixed resolution of 256 by 256 pixels.
  
Returns: data, label
************
__len__
Returns: the length of the loaded dataset.
'''
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split="train", resolution=256):
        # Loads the dataset that needs to be transformed
        self.dataset = dataset[split]
        self.resolution = resolution

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


# import requests
# TOKEN="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"
# headers = {"Authorization": f"Bearer {TOKEN}"}
# API_URL = "https://datasets-server.huggingface.co/rows?dataset=ibm/duorc&config=SelfRC&split=train&offset=150&length=10"
# def query():
#     response = requests.get(API_URL, headers=headers)
#     return response.json()
# data = query()


def load_dataset_from_huggingface(storage_dir, dataset_name="imagenet-1k"):
    # If the dataset is gated/private, make sure you have run the hugingface-cli login command
    dataset = load_dataset(dataset_name, cache_dir=storage_dir)
    dataset.save_to_disk(storage_dir)
    return dataset

if __name__ == "__main__":
    storage_dir = "/srv/storage/energyfl@storage1.toulouse.grid5000.fr/imagenet-1k"
    ds = load_from_disk(storage_dir)
    # .to_iterable_dataset().shuffle().with_format("torch")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_loader = DataLoader(
        ds["train"].with_transform(train_transforms).with_format("torch"), batch_size=4, collate_fn=collate_fn
        )
    
    i=0
    for images, labels in train_loader:
        print(images[0])
        print(labels[1])
        i+=1
        if i>5:
            break