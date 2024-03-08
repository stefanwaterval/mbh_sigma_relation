# import packages
import torch
from platform import release
from unittest import removeResult
import torch.nn as nn
from torch import flatten
from torch.utils.data import Dataset
import pandas as pd
  
class MyDataset(Dataset):
 
  def __init__(self, df, target_column, error_column):
    x = df.drop([target_column, error_column], axis='columns').values  # Features
    y = df[target_column].values  # Target
    errors = df[error_column].values # Errors
    self.data = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(errors, dtype=torch.float32)
 
  def __len__(self):
    return len(self.data[1])
   
  def __getitem__(self, idx):
    return self.data[0][idx], self.data[1][idx], self.data[2][idx]

class OneLayerLinearNet(nn.Module):
    
    def __init__(self,num_features):
        """Simple linear NN

        Parameters
        ----------
        num_features : int
            Number of galaxy parameters that we will use
        """
        # Call parent constructor
        super(OneLayerLinearNet, self).__init__()
      
        # Initialize layer(s)
        self.layers = nn.Sequential(nn.Linear(in_features=num_features, out_features=1))
        
    def forward(self,x):
        return self.layers(x)
      
class MultiLayerPerceptron(nn.Module):
    
    def __init__(self,num_features):
        """Simple MLP

        Parameters
        ----------
        num_features : int
            Number of galaxy parameters that we will use
        """
        # Call parent constructor
        super(MultiLayerPerceptron, self).__init__()
      
        # Initialize layer(s)
        self.layers = nn.Sequential(nn.Linear(in_features=num_features, out_features=64),
                                    nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=32),
                                    nn.ReLU(),
                                    nn.Linear(in_features=32, out_features=16),
                                    nn.ReLU(),
                                    nn.Linear(in_features=16, out_features=8),
                                    nn.ReLU(),
                                    nn.Linear(in_features=8, out_features=1))
        
    def forward(self,x):
        return self.layers(x)

            
        
        
        
      
    
      
        


