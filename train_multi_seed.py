from  neuralnets import MyDataset, OneLayerLinearNet, MultiLayerPerceptron
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import time
import sys

from columns import columns

from train import load_and_preprocess, train
from train import INIT_LR, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT

'''Functions definition'''
def train_multi_seed(seed_split, seed_pytorch, epochs):
    seeds = range(seed_split)
    training_losses = np.zeros((len(seeds), epochs))
    validation_losses = np.zeros((len(seeds), epochs))
    for i, s in enumerate(seeds):
        print(f'[INFO] Training seed {i+1}')
        
        # Set random seet
        torch.manual_seed(seed_pytorch)
        
        # Load dataset and prepare the data for training
        train_data_loader, val_data_loader, num_features = load_and_preprocess('data/SMBH_Data_01_26_24.csv',
                                                                            columns, 
                                                                            'M_BH',
                                                                            'M_BH_std_sym',
                                                                            TRAIN_SPLIT,
                                                                            BATCH_SIZE,
                                                                            s)
        
        # Train the model
        H = train(MultiLayerPerceptron, num_features, train_data_loader, val_data_loader,
              nn.MSELoss(reduction='none'), optimizer=Adam, lr=INIT_LR, batch_size=BATCH_SIZE, epochs=epochs)

        training_losses[i] = H['train_loss']
        validation_losses[i] = H['val_loss']
        
        print(f'[INFO] Seed {i+1} training finished')
        
    return seeds, training_losses, validation_losses
        
if __name__ == '__main__':
    # Train the models
    seeds, training_losses, validation_losses = train_multi_seed(1000, 42, 50)
    
    np.savetxt('./loss/mlp_training_losses.txt', training_losses)
    np.savetxt('./loss/mlp_validation_losses.txt', validation_losses)
    
    ## plot the training loss
    #plt.style.use("ggplot")
    #plt.figure()
    #plt.title(f'Epochs=5, seeds={len(seeds)}')
    #plt.plot(seeds, validation_losses, label="val_loss")
    #plt.xlabel('Seed')
    #plt.ylabel('Validation Loss')
    #plt.show()