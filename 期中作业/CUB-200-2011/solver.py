import torch.nn as nn
import torch.optim as optim
from data import preprocess_data
from model import CUB_ResNet_18


# get the pretrained model
model = CUB_ResNet_18()

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = optim.SGD([
    {'params': model.conv1.parameters(), 'lr': 0.0001},  
    {'params': model.bn1.parameters(), 'lr': 0.0001},
    {'params': model.layer1.parameters(), 'lr': 0.0001},
    {'params': model.layer2.parameters(), 'lr': 0.0001},
    {'params': model.layer3.parameters(), 'lr': 0.0001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.fc.parameters()}
], lr=0.001, momentum=0.9)