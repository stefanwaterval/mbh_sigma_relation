import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from columns import columns  # Assuming this contains the feature names

TRAIN_SPLIT = 0.75

# Function to load and preprocess the data
def load_and_preprocess(filepath, columns, target_column, error_column, trainsplit, seed_split):
    print('[INFO] loading the dataset...')
    df = pd.read_csv(filepath, header=1)
    df = df[columns]
    df = df.dropna()

    print("[INFO] generating the train/validation split...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the DataFrame
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=trainsplit, random_state=seed_split)
    
    # Separate the error column from the rest of the dataset
    y_train_errors = X_train[[error_column]].copy()
    y_val_errors = X_val[[error_column]].copy()
    
    X_train = X_train.drop(columns=[error_column])
    X_val = X_val.drop(columns=[error_column])

    # Compute normalization statistics from the training set
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    Y_mean = y_train.mean(axis=0)
    Y_std = y_train.std(axis=0)

    print("[INFO] normalizing the dataset...")
    X_train_normalized = (X_train - X_mean) / X_std
    y_train_normalized = (y_train - Y_mean) / Y_std
    X_val_normalized = (X_val - X_mean) / X_std
    y_val_normalized = (y_val - Y_mean) / Y_std

    return X_train_normalized, X_val_normalized, y_train_normalized, y_val_normalized, y_train_errors, y_val_errors

# Function to train the Random Forest model
def train_random_forest(X_train, y_train, y_train_errors, n_estimators, max_depth, seed_training):
    print('[INFO] training the Random Forest model...')
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed_training)
    model.fit(X_train, y_train, sample_weight=1/y_train_errors.to_numpy().flatten()**2)
    return model

# Function to evaluate the model
def evaluate(model, X_val, y_val):
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    print(f'[INFO] Validation RMSE: {rmse:.4f}')
    return rmse

def train_multi_seed(max_seed_split):
    seeds = range(max_seed_split)
    rmse_all = np.zeros(len(seeds))
    
    for i, s in enumerate(seeds):
        print(f'[INFO] Training seed {i}')
        
        # Load dataset and prepare the data for training
        X_train, X_val, y_train, y_val, y_train_errors, y_val_errors = load_and_preprocess('data/SMBH_Data_01_26_24.csv',
                                                                                           columns, 
                                                                                           'M_BH',
                                                                                           'M_BH_std_sym',
                                                                                           TRAIN_SPLIT,
                                                                                           s)
        
        try:
             # Train the Random Forest model
            model = train_random_forest(X_train, y_train, y_train_errors, n_estimators=100, max_depth=None, seed_training=42)
            # Evaluate the model
            rmse = evaluate(model, X_val, y_val)
        except:
            print(f'[ERROR] Training seed {i} failed')
            rmse = np.nan
        
        rmse_all[i] = rmse
        
    return rmse_all

# Main script
if __name__ == '__main__':
    ## Load dataset and prepare the data for training
    #X_train, X_val, y_train, y_val, y_train_errors, y_val_errors = load_and_preprocess('data/SMBH_Data_01_26_24.csv',
    #                                                                                   columns, 
    #                                                                                   'M_BH',
    #                                                                                   'M_BH_std_sym',
    #                                                                                   TRAIN_SPLIT,
    #                                                                                   42)

    ## Train the Random Forest model
    #model = train_random_forest(X_train, y_train, y_train_errors, n_estimators=100, max_depth=None, seed_training=42)

    ## Evaluate the model
    #rmse = evaluate(model, X_val, y_val)
    
    # Multi-seed training
    rmse_all = train_multi_seed(1000)
    
    np.savetxt('./loss/rf_rmse.txt', rmse_all)

    sys.exit(0)
    # Feature importance
    feature_importances = model.feature_importances_
    plt.barh(columns[:-2], feature_importances)  # Assuming the last two columns are target and error
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.show()
