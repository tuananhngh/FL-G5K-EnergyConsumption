from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def convert_bn_to_gn(module, num_groups=64):
    if isinstance(module, nn.BatchNorm2d):
        num_channels = module.num_features
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
        else:
            return nn.GroupNorm(1, num_channels)
    else:
        for name, child_module in module.named_children():
            module.add_module(name, convert_bn_to_gn(child_module, num_groups))
        return module

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, stride=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1
        )
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(
            in_features=84, out_features=10
        )  # 10 classes in CIFAR-10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)  # reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def train(model, trainloader, epochs, optimizer, device, verbose=False):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss, total_sample, correct = 0.0, 0.0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_val = criterion(outputs, labels)
            loss_val.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss_val.item()
            total_sample += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader)
        epoch_acc = correct / total_sample
        # # Validation metrics
        # val_loss, val_acc = test(model, valloader, verbose=False)
        result = {
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
        }
        # if verbose:
        #     logging.info("Epoch: {} | Client : {} | Loss: {} | Accuracy: {}".format(epoch, client_id, val_loss, val_acc))
    return result

def train_constraints(model, 
                      trainloader, 
                      epochs, 
                      optimizer,
                      constraints,
                      device,
                      verbose=False):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss, total_sample, correct = 0.0,0.0,0.0
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(constraints)
            
            # Metrics
            epoch_loss += loss.item()
            total_sample += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader)
        epoch_acc = correct / len(trainloader.dataset) 
        results = {"train_loss":epoch_loss, "train_acc":epoch_acc}
    return results

def test(model, testloader,verbose=False):
    criterion = nn.CrossEntropyLoss()
    #model.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss, total_sample, correct = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total_sample += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        avg_loss = val_loss/len(testloader)
        accuracy = correct/total_sample
        if verbose:
            logging.info("Loss: {} | Accuracy: {}".format(avg_loss, accuracy))
    return avg_loss, accuracy

def check_device(neural_net):
    is_model_on_gpu = next(neural_net.parameters()).is_cuda
    if is_model_on_gpu:
        print("Model is on GPU")
    else:
        print("Model is on CPU")