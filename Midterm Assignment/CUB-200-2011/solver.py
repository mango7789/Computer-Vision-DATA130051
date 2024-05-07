from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data import preprocess_data
from model import CUB_ResNet_18

# # set the working directory
# os.chdir('./Midterm Assignment/CUB-200-2011')

# load the dataset
train_loader, test_loader = preprocess_data()

# get the pretrained model
model = CUB_ResNet_18()

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimizer
optimizer = optim.SGD([
        {'params': model.resnet18.conv1.parameters(), 'lr': 0.0001},  
        {'params': model.resnet18.bn1.parameters(), 'lr': 0.0001},
        {'params': model.resnet18.layer1.parameters(), 'lr': 0.0001},
        {'params': model.resnet18.layer2.parameters(), 'lr': 0.0001},
        {'params': model.resnet18.layer3.parameters(), 'lr': 0.0001},
        {'params': model.resnet18.layer4.parameters(), 'lr': 0.0001},
        {'params': model.resnet18.fc.parameters()}
    ], 
    lr=0.001, 
    momentum=0.9
)

# init the tensorboard
writer = SummaryWriter()

# train the model and visualize
def fine_tuning(iter: int=10, ):
    for epoch in tqdm(range(iter)):
        # train
        model.train()
        running_loss = 0.0
        samples = 0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / samples
        writer.add_scalar('Train/Loss', epoch_loss, epoch)

        # test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)

# close the tensorboard
fine_tuning()
writer.flush()
