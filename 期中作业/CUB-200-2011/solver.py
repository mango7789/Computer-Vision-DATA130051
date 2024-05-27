import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import preprocess_data
from model import CUB_ResNet_18
from tqdm import tqdm

def seed_everything(seed: int=None):
    """
    Set the random seed for the whole neural network.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data_model_criterion(data_dir: str, pretrain: bool=True) -> tuple:
    """
    Get the DataLoader, model and loss criterion.
    """
    # load the dataset
    train_loader, test_loader = preprocess_data(data_dir)

    # get the pretrained model
    model = CUB_ResNet_18(pretrain=pretrain)

    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    return train_loader, test_loader, model, criterion

def calculate_topk_correct(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> list[int]:
    """
    Computes the top-k correct samples for the specified values of k.

    Args:
    - output (torch.Tensor): The model predictions with shape (batch_size, num_classes).
    - target (torch.Tensor): The true labels with shape (batch_size, ).
    - topk (tuple): A tuple of integers specifying the top-k values to compute.

    Returns:
    - List of top-k correct samples for each value in topk.
    """
    maxk = max(topk)

    # get the indices of the top k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res

def train_resnet_with_cub(
    data_dir: str,
    num_epochs: list[int], 
    fine_tuning_lr: float=0.0001, 
    output_lr: float=0.001, 
    pretrain: bool=True, 
    save: bool=False,
    **kwargs: dict
) -> list[float]:
    """
    Train the modified ResNet-18 model using the CUB-200-2011 dataset and return the best accuracy.
    Some hyper-parameters can be modified here.
    
    Args:
    - data_dir: The stored directory of the dataset.
    - num_epochs: A list of number of training epochs.
    - fine_tuning_lr: Learning rate of the parameters outside the output layer, default is 0.0001.
    - output_lr: Learning rate of the parameters inside the output layer, default is 0.001.
    - pretrain: Boolean, whether the ResNet-18 model is pretrained or not. Default is True.
    - save: Boolean, whether the parameters of the best model will be save. Default is False.
    
    Return:
    - best_acc: The best validation accuracy list during the training process.
    """
    # set the random seed
    seed_everything(kwargs.pop('seed', 42))
    
    # get the dataset, model and loss criterion
    train_loader, test_loader, model, criterion = get_data_model_criterion(data_dir, pretrain)
    
    # move the model to CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # get the parameters of the model expect the last layer
    former_params = [p for name, p in model.resnet18.named_parameters() if 'fc' not in name]
    
    # pop the hyper-parameters from the kwargs dict
    momentum = kwargs.pop('momentum', 0.9)
    weight_decay = kwargs.pop('weight_decay', 1e-4)
        
    # define optimizer
    optimizer = optim.SGD([
                {'params': former_params, 'lr': fine_tuning_lr, 'weight_decay': weight_decay},
                {'params': model.resnet18.fc.parameters(), 'lr': output_lr, 'weight_decay': weight_decay}
            ], momentum=momentum
        )
    
    # scheduler step size and gamma
    step_size = kwargs.pop('step', 30)
    gamma = kwargs.pop('gamma', 0.1)

    # custom step scheduler
    def custom_step_scheduler(optimizer: optim, epoch: int, step_size: int, gamma: float):
        """
        Decay the learning rate of the second parameter group by gamma every step_size epochs.
        """
        if epoch % step_size == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
    
    # init the tensorboard
    tensorboard_name = "/kaggle/working/Fine_Tuning_With_Pretrain"
    if len(num_epochs) != 1:
        tensorboard_name = "/kaggle/working/Full_Train"
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
        samples = 0
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
        
        # learning rate decay
        custom_step_scheduler(optimizer, epoch, step_size, gamma)
        
        train_loss = running_loss / samples
        print("[Epoch {:>2} / {:>2}], Training loss is {:>8.6f}".format(epoch + 1, max_num_epoch, train_loss))

        # test
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        samples = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                top1, top5 = calculate_topk_correct(outputs, labels, topk=(1, 5))
                correct_top1 += top1
                correct_top5 += top5
                samples += labels.size(0)
                
                running_loss += criterion(outputs, labels).item() * inputs.size(0)

        # add loss and accuracy to tensorboard
        test_loss = running_loss / samples
        writer.add_scalars('Loss', {'Train': train_loss, 'Valid': test_loss}, epoch + 1)
        accuracy_top1 = correct_top1 / samples
        accuracy_top5 = correct_top5 / samples
        writer.add_scalars(
            'Valid Accuracy', 
            {
                'Top1': accuracy_top1,
                'Top5': accuracy_top5,
            },
            epoch + 1
            
        )
        
        print("[Epoch {:>2} / {:>2}], Validation loss is {:>8.6f}, Top-5 accuracy is {:>8.6f}, Top-1 accuracy is {:>8.6f}".format(
            epoch + 1, max_num_epoch, test_loss, accuracy_top5, accuracy_top1
        ))
        
        # update the best accuracy and save the model if it improves
        if accuracy_top1 > best_acc:
            best_acc = accuracy_top1
            if save:
                torch.save(model.state_dict(), 'resnet18_cub.pth')
            
        if epoch + 1 == num_epochs[count]:
            store_best_acc[count] = best_acc
            count += 1

    # close the tensorboard
    writer.close()
    
    return store_best_acc

def test_resnet_with_cub(data_dir: str, path: str):
    """
    Test the trained model on the CUB dataset.
    
    Args:
    - data_dir: The stored directory of the dataset.
    - path: Path to the .pth file. 
    """
    # get the dataset, model and loss criterion
    train_loader, test_loader, model, _ = get_data_model_criterion(data_dir)
    
    # move the model to CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load the trained model
    trained_state_dict = torch.load(path, map_location=device)
    new_trained_state_dict = {}
    for key, value in trained_state_dict.items():
        if key.startswith("resnet18."):
            new_trained_state_dict[key[len("resnet18."):]] = value
        else:
            new_trained_state_dict[key] = value
    model.resnet18.load_state_dict(new_trained_state_dict)
    
    def dataset_accuracy(model: CUB_ResNet_18, data_loader: DataLoader, type: str):
        """
        Compute the accuracy based on the given model and dataset.
        
        Args:
        - model: The ResNet-18 model on the CUB dataset.
        - data_loader: The train/test dataloader.
        - type: The type of the computation of accuracy, should be in ['train', 'test'].
        """
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                top1, top5 = calculate_topk_correct(outputs, labels, topk=(1, 5))
                correct_top1 += top1
                correct_top5 += top5
                samples += labels.size(0)
                
        # compute accuracy
        accuracy_top1 = correct_top1 / samples
        accuracy_top5 = correct_top5 / samples
    
        # print the accuracy
        print("For the best model on the CUB dataset, Top-5 {:>5} accuracy is {:>8.6f}, Top-1 {:>5} accuracy is {:>8.6f}".format(
            type, accuracy_top5, type, accuracy_top1
        ))

    # dataset_accuracy(model, train_loader, 'train')
    dataset_accuracy(model, test_loader, 'test')