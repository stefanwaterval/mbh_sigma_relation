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

'''Functions definition'''
# Function to load and preprocess the data
def load_and_preprocess(filepath, columns, target_column, error_column, trainsplit, batch_size, seed):
    print('[INFO] loading the dataset...')
    # Load dataset
    df = pd.read_csv(filepath, header=1)
    df = df[columns]
    df = df.dropna()

    print("[INFO] generating the train/validation split...")
    # Split the DataFrame
    train_df = df.sample(frac=trainsplit, random_state=seed)
    val_df = df.drop(train_df.index)
    
    # Separate the error column from the rest of the dataset
    train_errors = train_df[[error_column]].copy()
    val_errors = val_df[[error_column]].copy()
    
    # Drop the error column from the main dataframes
    train_df = train_df.drop(columns=[error_column])
    val_df = val_df.drop(columns=[error_column])

    # Compute normalization statistics from the training set
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)

    print("[INFO] normalizing train and validation sets...")
    # Normalize training data
    train_df_normalized = (train_df - train_mean) / train_std

    # Normalize validation data using training statistics
    val_df_normalized = (val_df - train_mean) / train_std
    
    # Concatenate the error column back to the normalized dataframes
    train_df_normalized = pd.concat([train_df_normalized, train_errors], axis=1)
    val_df_normalized = pd.concat([val_df_normalized, val_errors], axis=1)

    train_dataset = MyDataset(train_df_normalized, target_column=target_column, error_column=error_column)
    val_dataset = MyDataset(val_df_normalized, target_column=target_column, error_column=error_column)

    # Initialize the train, validation, and test data loaders
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Get number of features
    num_features = len(columns) - 2 # Target column is included in columns
    
    return train_data_loader, val_data_loader, num_features

# Function to train the model
def train(model, num_features, train_data_loader, val_data_loader, loss_func,
          optimizer, lr, batch_size, epochs):
    # Set the device we will be using to train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate steps per epoch for training and validation set
    train_steps = (len(train_data_loader.dataset) + batch_size - 1) // batch_size
    val_steps = (len(val_data_loader.dataset) + batch_size -1) // batch_size
    
    # Initialize the model
    model = model(num_features=num_features).to(device)
    
    # Initialize optimizer and loss function
    opt = optimizer(model.parameters(), lr=lr, weight_decay=0.001)
    
    # Define the scheduler
    #scheduler = StepLR(opt, step_size=10, gamma=0.1)  # Decay LR by a factor of 0.1 every 30 epochs
    scheduler = ExponentialLR(opt, gamma=0.9)  # Decay LR by a factor of 0.9 every 30 epochs
    
    # Initialize dictionary to store train history
    H = {'train_loss': [],
         'val_loss': []}

    # Start training
    print('[INFO] training the network...')
    tick = time.time()

    for e in range(epochs):
        # Set model in training mode
        model.train()

        # Initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        for (x, y, error) in train_data_loader:
            # Send input to device
            (x, y, error) = (x.to(device), y.to(device), error.to(device))

            # Perform a forward pass and calculate the training loss
            pred = model(x)
            weights = 1.0 / error**2
            loss = loss_func(pred, y)#.unsqueeze(1))
            weighted_loss = (weights * loss).sum() / weights.sum()

            # Zero out the gradients, perform the backpropagation step,
    		# and update the weights
            opt.zero_grad()
            weighted_loss.backward()
            opt.step()

            # Add the loss to the total training loss so far and
    		# calculate the number of correct predictions
            total_train_loss += weighted_loss
            
        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Perform validation
        # Switch off autograds for validation
        with torch.no_grad():
            # Set the model in evaluation mode
            model.eval()

            for (x , y, error) in val_data_loader:
                (x , y, error) = (x.to(device), y.to(device), error.to(device))

                pred = model(x)
                weights = 1.0 / error**2
                loss = loss_func(pred, y)
                weighted_loss = (weights * loss).sum() / weights.sum()
                total_val_loss += weighted_loss

        # calculate the average training and validation loss
        avg_train_loss = torch.sqrt(total_train_loss / train_steps)
        avg_val_loss = torch.sqrt(total_val_loss / val_steps)
    
    	# update our training history
        H["train_loss"].append(avg_train_loss.item())
        H["val_loss"].append(avg_val_loss.item())

    	# print the model training and validation information
        print(f"[INFO] EPOCH: {e+1}/{epochs}")
        print(f"Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}")

    # finish measuring how long training took
    tock = time.time()
    print(f"[INFO] total time taken to train the model: {tock-tick:.2f}s")
    
    # serialize the model to disk
    #print(f'[INFO] Saving model to disk')
    #torch.save(model.state_dict(), 'trained_models/mlp_columns.pth')
    
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)
    
    return H

'''Constants definition'''
# Define training parameters
INIT_LR = 1e-3
BATCH_SIZE = 1

# Define the train and validation splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

'''Main'''
if __name__ == '__main__':
    # Set the random seed for reproducible results
    seed = 42
    torch.manual_seed(seed)
    #np.random.seed(42)

    # Load dataset and prepare the data for training
    train_data_loader, val_data_loader, num_features = load_and_preprocess('data/SMBH_Data_01_26_24.csv',
                                                                            columns, 
                                                                            'M_BH',
                                                                            'M_BH_std_sym',
                                                                            TRAIN_SPLIT,
                                                                            BATCH_SIZE,
                                                                            seed)

    # Train the models
    #H = train(OneLayerLinearNet, num_features, train_data_loader, val_data_loader,
    #                   nn.MSELoss(), optimizer=Adam, lr=INIT_LR, batch_size=BATCH_SIZE, epochs=2000)
    H = train(MultiLayerPerceptron, num_features, train_data_loader, val_data_loader,
              nn.MSELoss(reduction='none'), optimizer=Adam, lr=INIT_LR, batch_size=BATCH_SIZE, epochs=20)

    # plot the training loss
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    #plt.savefig(args["plot"])
    plt.show()
