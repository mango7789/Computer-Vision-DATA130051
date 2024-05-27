import os
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

def load_VOC(batch_size: int=64, shuffle: bool=True):

    # convert PIL image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # download = not os.path.exists('./data')
    download = True
    
    # load dataset
    train_dataset = VOCDetection(root='./data', year='2007', image_set='train', transform=transform, download=download)
    test_dataset = VOCDetection(root='./data', year='2007', image_set='val', transform=transform, download=download)
    
    # get the dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

load_VOC()