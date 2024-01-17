from typing import OrderedDict
from sympy import Ne
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenet import mobilenet_v3_small
from torchinfo import summary
# Basic CNN Model
class Net(nn.Module):
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


    
# ResNet18 Model

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet,self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block,512,num_blocks[3],stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out
    
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

# resnet18 = ResNet18()
# basic = Net()

# sample = torch.randn(2,3,32,32)
# model = mobilenet_v3_small(num_classes=10)

# summary(model, (60,3, 32, 32), device='cpu')
# summary(resnet18, (60, 3, 32, 32), device='cpu')
# summary(basic, (60, 3, 32, 32), device='cpu')


# from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
# import utils

# params = utils.get_parameters(resnet18)
# params_out = ndarrays_to_parameters(params)

# params_in = parameters_to_ndarrays(params_out)

# keys = [name for name,val in resnet182.state_dict().items()]
# params_dict = zip(keys, params)
# state_dict = OrderedDict({k:torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0]) for k,v in params_dict})

# resnet182.load_state_dict(state_dict, strict = True)