import os
# set the working directory
try:
    os.chdir('./Midterm Assignment/CUB-200-2011')
except:
    pass
# set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from itertools import product
from solver import train_resnet_with_cub

# set hyper-parameters here
num_epochs = [10, 15, 20]
fine_tuning_lrs = [0.0001, 0.0005, 0.01]
output_lrs = [0.01, 0.02, 0.04, 0.06]

configurations = list(product(num_epochs, fine_tuning_lrs, output_lrs))

best_accs = []

# train with the pretrained model
for config in configurations:
    curr_best_acc = train_resnet_with_cub(*config)
    best_accs.append(curr_best_acc)
    
# write the results into a txt file
with open('best_accuracy.txt', 'w') as f:
    for config, accuracy in zip(configurations, best_accs):
        f.write(f"Configuration: {config}, Accuracy: {accuracy}\n")
