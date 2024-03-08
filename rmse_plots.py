import sys

import numpy as np

import matplotlib.pyplot as plt

# Load data
mlp_train_losses = np.loadtxt('./loss/mlp_training_losses.txt')
mlp_val_losses = np.loadtxt('./loss/mlp_validation_losses.txt')
rf_rmse = np.loadtxt('./loss/rf_rmse.txt')

# Prepare data
## Calculate fractions of NaNs
mlp_train_nan_frac = np.isnan(mlp_train_losses).all(axis=1).sum() / mlp_train_losses.shape[0]
mlp_val_nan_frac = np.isnan(mlp_val_losses).all(axis=1).sum() / mlp_val_losses.shape[0]
rf_nan_frac = np.isnan(rf_rmse).sum() / len(rf_rmse)

## Remove NaNs
mlp_train_losses_wo_nan = mlp_train_losses[~np.isnan(mlp_train_losses).all(axis=1)]
mlp_val_losses_wo_nan = mlp_val_losses[~np.isnan(mlp_val_losses).all(axis=1)]
rf_rmse_wo_nan = rf_rmse[~np.isnan(rf_rmse)]

## Calculate 16, 50, 84 percentiles for MLP
mlp_train_16 = np.percentile(mlp_train_losses_wo_nan, 16, axis=0)
mlp_train_50 = np.percentile(mlp_train_losses_wo_nan, 50, axis=0)
mlp_train_84 = np.percentile(mlp_train_losses_wo_nan, 84, axis=0)

mlp_val_16 = np.percentile(mlp_val_losses_wo_nan, 16, axis=0)
mlp_val_50 = np.percentile(mlp_val_losses_wo_nan, 50, axis=0)
mlp_val_84 = np.percentile(mlp_val_losses_wo_nan, 84, axis=0)

# Plots

## MLP all seeds vs epoch
fig1, ax1 = plt.subplots(1, 3, figsize=(15, 5))
epochs = np.arange(1, 51)
for i in range(len(mlp_train_losses_wo_nan)):
    ax1[0].plot(epochs, mlp_train_losses_wo_nan[i], 'r', alpha=0.2)
    ax1[1].plot(epochs, mlp_val_losses_wo_nan[i], 'b', alpha=0.2)

ax1[0].plot(epochs, mlp_train_50, 'k', ls='-', label='Training Loss')
ax1[1].plot(epochs, mlp_val_50, 'k', ls='-', label='Validation Loss')

ax1[2].plot(epochs, mlp_train_50, 'r', ls='-', label='Training Loss')
ax1[2].fill_between(epochs, mlp_train_16, mlp_train_84, color='r', alpha=0.4)
ax1[2].plot(epochs, mlp_val_50, 'b', ls='-', label='Validation Loss')
ax1[2].fill_between(epochs, mlp_val_16, mlp_val_84, color='b', alpha=0.4)

ax1[0].text(10, 1.2, f'NaN fraction: {mlp_train_nan_frac:.2f}')
ax1[1].text(10, 1.2, f'NaN fraction: {mlp_val_nan_frac:.2f}')

for ax in ax1.flatten():
    ax.set_xlabel('Epoch')
    ax.set_ylim(0, 1.3)
    ax.legend(loc='best')
    
ax1[0].set_ylabel('RMSE Loss')

plt.tight_layout()
plt.show()

## MLP vs RF
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 5))
seeds = range(1000)
ax2.plot(seeds, rf_rmse, 'k', marker='o', ls='', label='Random Forest')
ax2.plot(seeds, mlp_train_losses[:,-1], 'r', marker='o', ls='', label='MLP Training')
ax2.plot(seeds, mlp_val_losses[:,-1], 'b', marker='o', ls='', label='MLP Validation')

ax2.set_xlabel('Data split seed')
ax2.set_ylabel('RMSE')

ax2.set_ylim(0, 1.2)

ax2.legend(loc='best')

ax2.text(1, 1, f'RF NaN fraction: {rf_nan_frac:.2f}')
ax2.text(1, 0.9, f'MLP NaN fraction: {mlp_train_nan_frac:.2f}')

plt.tight_layout()
plt.show()



