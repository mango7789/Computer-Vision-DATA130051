import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def preprocess_data(batch_size: int=128) -> tuple[DataLoader, DataLoader]:
    """
    Preprocess the CUB-200-2011 dataset and return the train and test batches.
    
    Args:
    - batch_size: The number of samples in one batch, default is 128.
    """
    
    # resize and normalize the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_dir = 'CUB_200_2011_dataset'

    # TODO: Write a class to get the train and test images/labels
    # load the dataset and extract the train/test Dataloader
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader