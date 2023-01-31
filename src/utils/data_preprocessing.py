import argparse
import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from src.utils.domain_split import hyperplane_split
from src.utils import data_loaders


def prepare_data(data_path: str, args: argparse.Namespace):
    """
    Conducts all the required data preprocessing to prepare the data
    for the next step of the pipeline. Customized actions for each dataset.
    """

    # Extract arguments
    seed = args.seed
    device = args.device
    verbose = args.verbose
    dataset = args.dataset
    batch_size = args.batch_size
    
    if dataset == 'airfoil':
        
        # Read data
        df = pd.read_csv(data_path)
        D = df.values

        # Split domain
        dom_idx, OOD_idx = hyperplane_split(D, OOD_size=0.2, verbose=verbose, seed=seed)

        # Minmax scaling for better network performace
        scaler = MinMaxScaler()
        D = df[dom_idx].values
        D = scaler.fit_transform(D)

        # Split in-domain data to data and labels
        X, y = D[:,:-1], D[:,-1]

        # Split to train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Separate OOD data
        OOD = df[OOD_idx].values
        OOD = scaler.transform(OOD)         # need to also transform it to be compatible with the NN.
        X_OOD, y_OOD = OOD[:,:-1], OOD[:,-1]

        # Transformation required for regression problem
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        y_OOD = y_OOD.reshape(-1, 1)

        # Create tensor datasets
        train_dataset = data_loaders.create_torch_dataset(X_train, y_train, to_LongTensor=False)
        test_dataset = data_loaders.create_torch_dataset(X_test, y_test, to_LongTensor=False)
        OOD_dataset = data_loaders.create_torch_dataset(X_OOD, y_OOD, to_LongTensor=False)

        # Data loaders on gpu
        train_loader = data_loaders.create_loader(train_dataset, batch_size, device)
        test_loader = data_loaders.create_loader(test_dataset, batch_size, device)
        OOD_loader = data_loaders.create_loader(OOD_dataset, batch_size, device)

        n_data = X_train.shape[0]
        n_features = X_train.shape[1]
        n_classes = 1 # regression

        return n_data, n_features, n_classes, train_loader, test_loader, OOD_loader

    
    elif dataset == 'boston':
        
        # Read data
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        df = pd.read_csv(data_path, header=None, delimiter=r"\s+", names=column_names)
        df = df.drop(columns=['CRIM', 'ZN', 'CHAS', 'RAD', 'B'])
        D = df.values

        # Split domain
        dom_idx, OOD_idx = hyperplane_split(D, OOD_size=0.2, verbose=verbose, seed=seed)

        # Minmax scaling for better network performace
        scaler = MinMaxScaler()
        D = df[dom_idx].values
        D = scaler.fit_transform(D)

        # Split in-domain data to data and labels
        X, y = D[:,:-1], D[:,-1]

        # Split to train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Separate OOD data
        OOD = df[OOD_idx].values
        OOD = scaler.transform(OOD)         # need to also transform it to be compatible with the NN.
        X_OOD, y_OOD = OOD[:,:-1], OOD[:,-1]

        # Transformation required for regression problem
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        y_OOD = y_OOD.reshape(-1, 1)

        # Create tensor datasets
        train_dataset = data_loaders.create_torch_dataset(X_train, y_train, to_LongTensor=False)
        test_dataset = data_loaders.create_torch_dataset(X_test, y_test, to_LongTensor=False)
        OOD_dataset = data_loaders.create_torch_dataset(X_OOD, y_OOD, to_LongTensor=False)

        # Data loaders on gpu
        train_loader = data_loaders.create_loader(train_dataset, batch_size, device)
        test_loader = data_loaders.create_loader(test_dataset, batch_size, device)
        OOD_loader = data_loaders.create_loader(OOD_dataset, batch_size, device)

        n_data = X_train.shape[0]
        n_features = X_train.shape[1]
        n_classes = 1 # regression

        return n_data, n_features, n_classes, train_loader, test_loader, OOD_loader       
    
    
    elif dataset == 'concrete':
    
        # Read data
        df = pd.read_csv(data_path, index_col=0)
        D = df.values

        # Split domain
        dom_idx, OOD_idx = hyperplane_split(D, OOD_size=0.2, verbose=verbose, seed=seed)

        # Minmax scaling for better network performace
        scaler = MinMaxScaler()
        D = df[dom_idx].values
        D = scaler.fit_transform(D)

        # Split in-domain data to data and labels
        X, y = D[:,:-1], D[:,-1]

        # Split to train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Separate OOD data
        OOD = df[OOD_idx].values
        OOD = scaler.transform(OOD)         # need to also transform it to be compatible with the NN.
        X_OOD, y_OOD = OOD[:,:-1], OOD[:,-1]

        # Transformation required for regression problem
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        y_OOD = y_OOD.reshape(-1, 1)

        # Create tensor datasets
        train_dataset = data_loaders.create_torch_dataset(X_train, y_train, to_LongTensor=False)
        test_dataset = data_loaders.create_torch_dataset(X_test, y_test, to_LongTensor=False)
        OOD_dataset = data_loaders.create_torch_dataset(X_OOD, y_OOD, to_LongTensor=False)

        # Data loaders on gpu
        train_loader = data_loaders.create_loader(train_dataset, batch_size, device)
        test_loader = data_loaders.create_loader(test_dataset, batch_size, device)
        OOD_loader = data_loaders.create_loader(OOD_dataset, batch_size, device)

        n_data = X_train.shape[0]
        n_features = X_train.shape[1]
        n_classes = 1 # regression

        return n_data, n_features, n_classes, train_loader, test_loader, OOD_loader    
    
    
    
    
    
    
    
    
    
    
    
    
    