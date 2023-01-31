from functools import partial

import model
import evaluate
from utils import data


import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import torchmetrics as tm

import pyro
import pyro.distributions as dist

import tyxe
pyro.set_rng_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data transformations
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

# Download MNIST data
train_data = datasets.MNIST(root="data", train=True, transform=train_transform, download=True)
test_data = datasets.MNIST(root="data", train=False, transform=test_transform)

NUM_CLASSES = 10



train_size = 1000
test_size = 1000
train_batch_size = 100
test_batch_size = 100

train_data = data.create_subset(train_data, train_size)
test_data = data.create_subset(test_data, test_size)

train_loader = data.create_loader(train_data, train_batch_size, device=device)
test_loader = data.create_loader(test_data, test_batch_size, device=device)

N_EPOCHS = 100
LR = 1e-2


# VI
net = model.CNN_MNIST().to(device)
name = ""
# prior, note that weights and bias have the same sd
prior = tyxe.priors.IIDPrior(
    dist.Normal(
        torch.tensor(0, device=device, dtype=torch.float),
        torch.tensor(1, device=device, dtype=torch.float),
    ),
)
prior_name = ", IID prior"

# likelihood
obs_model = tyxe.likelihoods.Categorical(train_size)
likelihood_name = ", Categorical likelihood"

# mean-field VI
guide_builder = partial(tyxe.guides.AutoNormal, init_scale=0.01)
guide_name = "Mean field"

# bnn
bnn = tyxe.VariationalBNN(net, prior, obs_model, guide_builder)
bnn.name = guide_name + " Variational BNN" + likelihood_name + prior_name

pyro.clear_param_store()

# Adam optimizer
optim = pyro.optim.Adam({"lr": LR})

# callback function to keep track of ELBOs
elbos = []
def callback(bnn, i, e):
    elbos.append(e)

# Fit the model
fit = bnn.fit(train_loader, optim, N_EPOCHS, callback, device=device)


plt.figure(figsize=(9,6))
plt.plot(elbos)
plt.grid()
plt.xlabel("Iterations", fontsize=20)
plt.ylabel("ELBO", fontsize=20)
plt.title("ELBO plot", fontsize=20)
plt.show()


# Predict
posterior_samples = 100        # how many posterior "networks" to use for prediction
evaluation = evaluate.evaluate(bnn, test_loader, posterior_samples=posterior_samples,
                               device=device, dataset="MNIST", num_classes=10, rounded=4)

for key, value in evaluation.items():
    print(key + str(":"), value)
