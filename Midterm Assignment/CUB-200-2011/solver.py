from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data import preprocess_data
from model import CUB_ResNet_18

def get_data_model_criterion(pretrain: bool=True):
    """
    Get the DataLoader, model and loss criterion.
    """
    # load the dataset
    train_loader, test_loader = preprocess_data()

    # get the pretrained model
    model = CUB_ResNet_18()

    # define loss function
    criterion = nn.CrossEntropyLoss(pretrain=pretrain)
    
    return train_loader, test_loader, model, criterion

def train_resnet_with_cub(num_epoch: int=10, fine_tuning_lr: float=0.0001, output_lr: float=0.001, pretrain: bool=True):
    """
    Train the modified ResNet-18 model using the CUB-200-2011 dataset. Some hyper-parameters can be modified here.
    
    Args:
    - num_epoch: The number of training epochs, default is 10.
    - fine_tuning_lr: Learning rate of the parameters outside the output layer, default is 0.0001.
    - output_lr: Learning rate of the parameters inside the output layer, default is 0.001.
    - pretrain: Boolean, whether the ResNet-18 model is pretrained or not. Default is True.
    """
    # get the dataset, model and loss criterion
    train_loader, test_loader, model, criterion = get_data_model_criterion(pretrain)
    
    # define optimizer
    optimizer = optim.SGD([
                {'params': model.resnet18.conv1.parameters(), 'lr': fine_tuning_lr},  
                {'params': model.resnet18.bn1.parameters(), 'lr': fine_tuning_lr},
                {'params': model.resnet18.layer1.parameters(), 'lr': fine_tuning_lr},
                {'params': model.resnet18.layer2.parameters(), 'lr': fine_tuning_lr},
                {'params': model.resnet18.layer3.parameters(), 'lr': fine_tuning_lr},
                {'params': model.resnet18.layer4.parameters(), 'lr': fine_tuning_lr},
                {'params': model.resnet18.fc.parameters()}
            ], 
            lr=output_lr, 
        )
    
    # init the tensorboard
    writer = SummaryWriter("Fine_Tuning_With_Pretrain", comment="-{}-{}-{}".format(num_epoch, fine_tuning_lr, output_lr))\
                    if pretrain else SummaryWriter("Fine_Tuning_Random_Init", comment="-{}-{}-{}".format(num_epoch, fine_tuning_lr, output_lr))
    
    # iterate
    for epoch in tqdm(range(num_epoch)):
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
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                running_loss += criterion(outputs, labels).item() * inputs.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        writer.add_scalar('Validation/Loss', epoch_loss, epoch)
        accuracy = correct / total
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)

    # close the tensorboard
    writer.close()
