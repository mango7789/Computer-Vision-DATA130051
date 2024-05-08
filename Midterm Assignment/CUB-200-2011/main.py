from itertools import product
from solver import train_resnet_with_cub

# set hyper-parameters here
num_epochs = []
fine_tuning_lrs = []
output_lrs = []

configurations = list(product(num_epochs, fine_tuning_lrs, output_lrs))

# train with the pretrained model
for config in configurations:
    train_resnet_with_cub(*config)
