import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from data import preprocess_data
from model import CUB_ResNet_18

def seed_everything(seed: int=509):
    """
    Set the random seed for the whole neural network.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data_model_criterion(pretrain: bool=True) -> tuple:
    """
    Get the DataLoader, model and loss criterion.
    """
    # load the dataset
    train_loader, test_loader = preprocess_data()

    # get the pretrained model
    model = CUB_ResNet_18(pretrain=pretrain)

    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    return train_loader, test_loader, model, criterion

def train_resnet_with_cub(
    num_epochs: list[int], 
    fine_tuning_lr: float=0.0001, 
    output_lr: float=0.001, 
    pretrain: bool=True, 
    **kwargs: dict
) -> list[float]:
    """
    Train the modified ResNet-18 model using the CUB-200-2011 dataset and return the best accuracy.
    Some hyper-parameters can be modified here.
    
    Args:
    - num_epochs: A list of number of training epochs.
    - fine_tuning_lr: Learning rate of the parameters outside the output layer, default is 0.0001.
    - output_lr: Learning rate of the parameters inside the output layer, default is 0.001.
    - pretrain: Boolean, whether the ResNet-18 model is pretrained or not. Default is True.
    
    Return:
    - best_acc: The best validation accuracy list during the training process.
    """
    # set the random seed
    seed_everything(kwargs.pop('seed', 509))
    
    # get the dataset, model and loss criterion
    train_loader, test_loader, model, criterion = get_data_model_criterion(pretrain)
    
    # move the model to CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # get the parameters of the model expect the last layer
    former_params = [p for name, p in model.resnet18.named_parameters() if 'fc' not in name]
    
    # pop the hyper-parameters from the kwargs dict
    momentum = kwargs.pop('momentum', 0.9)
        
    # define optimizer
    optimizer = optim.SGD([
                {'params': former_params, 'lr': fine_tuning_lr},
                {'params': model.resnet18.fc.parameters()}
            ], lr=output_lr, momentum=momentum
        )
    
    # scheduler step size and gamma
    step_size = kwargs.pop('step', 10)
    gamma = kwargs.pop('gamma', 0.5)

    # custom step scheduler
    def custom_step_scheduler(optimizer: optim, epoch: int, step_size: int, gamma: float):
        """
        Decay the learning rate of the second parameter group by gamma every step_size epochs.
        """
        if epoch % step_size == 0 and epoch > 0:
            for index, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] *= gamma
    
    # init the tensorboard
    tensorboard_name = "Fine_Tuning_With_Pretrain" if pretrain else "Fine_Tuning_Random_Initialize"
    writer = SummaryWriter(tensorboard_name, comment="-{}-{}".format(fine_tuning_lr, output_lr))
        
    # best accuracy
    best_acc = 0.0
    store_best_acc, count = [0 for _ in range(len(num_epochs))], 0
    max_num_epoch = max(num_epochs)

    print("=" * 70)
    print("Training with configuration ({:>7.5f}, {:>7.5f})".format(fine_tuning_lr, output_lr))
    
    # iterate
    for epoch in range(max_num_epoch):
        # train
        model.train()
        running_loss = 0.0
        samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
        
        custom_step_scheduler(optimizer, epoch, step_size, gamma)
        train_loss = running_loss / samples
        print("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, max_num_epoch, train_loss))

        # test
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                running_loss += criterion(outputs, labels).item() * inputs.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / total
        writer.add_scalars('Loss', {'Train': train_loss, 'Valid': test_loss}, epoch + 1)
        accuracy = correct / total
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        print("[Epoch {:>2} / {:>2}], Validation loss is {:>8.6f}, Validation accuracy is {:>8.6f}".format(
            epoch + 1, max_num_epoch, test_loss, accuracy
        ))
        best_acc = max(best_acc, accuracy)
        
        if epoch + 1 == num_epochs[count]:
            store_best_acc[count] = best_acc
            count += 1

    # close the tensorboard
    writer.close()
    
    return store_best_acc