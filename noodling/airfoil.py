import os
# path = os.getcwd()
# print("path =", path)

#if path.split(sep="\\")[-1]!='InternshipMatakos':
#      os.chdir("..")

# os.chdir("..")

print("\npath =", os.getcwd())

from functools import partial


from src import evaluate
from src.evaluate import evaluate_regr
import utils
from utils import util
from utils import data_loaders
from utils.domain_split import hyperplane_split
from src.train_test import fit_regr
from src.train_test import test_regr
from src.model import MLP

import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.distributions.normal import Normal as norm
import torchmetrics as tm
import pyro
import pyro.distributions as dist
import tyxe

seed = 7141
npr.seed(seed)
torch.manual_seed(seed)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Read airfoil data
df = pd.read_csv("data/regression/airfoil.csv")
D = df.values

# Split domain
dom_idx, OOD_idx = hyperplane_split(D, OOD_size=0.2, verbose=1, seed=seed)

# Data processing
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

# Hyperparameters
n_features = X_train.shape[1]
n_hidden_1 = 100
n_hidden_2 = 30
n_epochs = 1000
learning_rate = 1e-4
weight_decay = 1e-5
batch_size_train = 64
batch_size_test = 64

# Transformation required for regression problem
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_OOD = y_OOD.reshape(-1, 1)

# Tensor datasets
train_dataset = data_loaders.create_torch_dataset(X_train, y_train, to_LongTensor=False)
test_dataset = data_loaders.create_torch_dataset(X_test, y_test, to_LongTensor=False)
OOD_dataset = data_loaders.create_torch_dataset(X_OOD, y_OOD, to_LongTensor=False)

# Data loaders on gpu
train_loader = data_loaders.create_loader(train_dataset, batch_size_train, device)
test_loader = data_loaders.create_loader(test_dataset, batch_size_test, device)
OOD_loader = data_loaders.create_loader(OOD_dataset, batch_size_test, device)

# BNN model
net = MLP(n_features, 1, [30, ], 'tanh').to(device)
prior = tyxe.priors.IIDPrior(dist.Normal(
        torch.tensor(0, device=device, dtype=torch.float),
        torch.tensor(1, device=device, dtype=torch.float)))
prior_name = ", IID prior"
likelihood = tyxe.likelihoods.HomoskedasticGaussian(len(X_train), scale=1e-1)
likelihood_name = ", Homoskedastic Gaussian likelihood"
# kernel = partial(pyro.infer.mcmc.HMC, step_size=1e-3, num_steps=20, target_accept_prob=0.7)
# kernel_name = 'HMC'
kernel = partial(pyro.infer.mcmc.NUTS)
kernel_name = "NUTS"
loss_function = nn.MSELoss()
bnn = tyxe.bnn.MCMC_BNN(net, prior, likelihood, kernel)
bnn.name = kernel_name + ", MCMC BNN" + likelihood_name + prior_name

# Fit
pyro.clear_param_store()
bnn.fit(train_loader, 1000, warmup_steps=500)

# Predict
posterior_samples = 100
train_evaluation = evaluate_regr(bnn, train_loader, loss_function, posterior_samples=posterior_samples, rounded=10,
                           dataset="airfoil, Train data", device=device)
test_evaluation = evaluate_regr(bnn, test_loader, loss_function, posterior_samples=posterior_samples, rounded=10,
                           dataset="airfoil, Test data", device=device)
OOD_evaluation = evaluate_regr(bnn, OOD_loader, loss_function, posterior_samples=posterior_samples, rounded=10,
                           dataset="airfoil, OOD data", device=device)


for key,  value in train_evaluation.items():
    print(key + str(":"), value)

for key,  value in test_evaluation.items():
    print(key + str(":"), value)

for key,  value in OOD_evaluation.items():
    print(key + str(":"), value)


# if __name__ == "__main__":
