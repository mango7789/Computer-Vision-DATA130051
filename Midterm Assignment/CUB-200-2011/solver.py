import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(509) # set the random seed

from tqdm import tqdm
from data import preprocess_data
from model import CUB_ResNet_18


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


def train_resnet_with_cub(num_epochs: list[int], fine_tuning_lr: float=0.0001, output_lr: float=0.001, pretrain: bool=True, **kwargs) -> list[float]:
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
    
    # init the tensorboard
    tensorboard_name = "Fine_Tuning_With_Pretrain" if pretrain else "Fine_Tuning_Random_Initialize"
    writer = SummaryWriter(tensorboard_name, comment="-{}-{}".format(fine_tuning_lr, output_lr))
        
    # best accuracy
    best_acc = 0.0
    store_best_acc, count = [0 for _ in range(len(num_epochs))], 0
    max_num_epoch = max(num_epochs)

    print("=" * 70)
    print("Training with configuration ({:>6.4f}, {:>6.4f})".format(fine_tuning_lr, output_lr))
    
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
        
        epoch_loss = running_loss / samples
        print("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, max_num_epoch, epoch_loss))
        writer.add_scalar('Train/Loss', epoch_loss, epoch)

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

        epoch_loss = running_loss / total
        writer.add_scalar('Validation/Loss', epoch_loss, epoch)
        accuracy = correct / total
        writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        print("[Epoch {:>2} / {:>2}], Validation loss is {:>8.6f}, Validation accuracy is {:>8.6f}".format(
            epoch + 1, max_num_epoch, epoch_loss, accuracy
        ))
        best_acc = max(best_acc, accuracy)
        
        if epoch + 1 == num_epochs[count]:
            store_best_acc[count] = best_acc
            count += 1

    # close the tensorboard
    writer.close()
    
    return store_best_acc