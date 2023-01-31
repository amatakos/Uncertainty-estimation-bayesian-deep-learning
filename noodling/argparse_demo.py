import os
import argparse
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
from functools import partial
from laplace import Laplace, marglik_training
from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
import src
from src import evaluate
from src.evaluate import evaluate_regr
from src.utils import data_preprocessing
from src.utils.domain_split import hyperplane_split
from src.utils.data_paths import data_path
from src.utils.random_seeds import seeds
from src.utils.util import print_dict
from src.train_test import fit_regr
from src.train_test import test_regr
from src.model import MLP


# Set up argparse
parser = argparse.ArgumentParser(description='Argparse demo')

# Required arguments
# Dataset
parser.add_argument('-ds', '--dataset', type=str, metavar='', required=True, 
                    help="Dataset name eg. 'boston'.")
# Inference
parser.add_argument('-i', '--inference', type=str, metavar='', required=True,
                    help="{{'MCMC', 'MFVI', 'laplace', 'ensemble'}}\nInference method for approximation of the posterior.")

# Optional arguments (with default settings)
# Location
parser.add_argument('-l', '--loc', type=str, metavar='', default='local', required=False, 
                    help="{{'local', 'cluster'}}, default='local' \nSpecify 'local' if running locally or 'cluster' if running on the cluster")
# Seed
#parser.add_argument('--seed', type=int, metavar='', default=0, required=False, 
#                    help="{{0: fixed, 1: random}}, default=0 \nRandom seed for reproducibility. 'fixed' uses a pretedetermined seed for each dataset that produces a 'good' domain split.")
# Device
parser.add_argument('-d', '--device', type=str, metavar='', default='cuda', required=False, 
                    help="{{'cuda', 'cpu'}}, default='cuda' \nWhether to use the CPU or GPU for computations")
# Verbose
parser.add_argument('-v', '--verbose', type=int, metavar='', default=0, required=False, 
                    help='{{0: minimal, 1: results, 2: results and intermediate steps}}, default=0 \nAmount of verbosity.')
# Network
parser.add_argument('-net', '--network', type=str, nargs='+', metavar='', default=[64, ], required=False, 
                    help='[...], default=[64,] \nNeural Network architecture, eg. -net 100 30 10 is 3-layered NN with 100, 30, 10 units.')
# Activation
parser.add_argument('-a', '--activation', type=str, metavar='', default='relu', required=False,
                    help="{{'relu', 'tanh', 'sigmoid'}}, default='relu' \nActivation function used in the network.")
# Batch size
parser.add_argument('-bs', '--batch_size', type=int, metavar='', default=128, required=False,
                    help='default=128 \nBatch size to use in the pytorch dataloaders.')
# Number of epochs
parser.add_argument('-ne', '--n_epochs', type=int, metavar='', default=300, required=False,
                    help='default=300 \nNumber of training epochs.')
# Early stopping 
parser.add_argument('-es', '--early_stopping', type=str, metavar='', default='dev', required=False,
                    help="{{'train', 'dev'}}, default='dev' \nWhether to apply early stopping during training phase. 'dev' checks performance on a dev set during each epoch.")
# Learning rate
parser.add_argument('-lr', '--learning_rate', type=float, metavar='', default=1e-4, required=False,
                    help="default=1e-4 \nLearning rate.")
# Weight decay
parser.add_argument('-wd', '--weight_decay', type=float, metavar='', default=1e-5, required=False,
                    help="default=1e-5 \nWeight decay for Adam optimizer. ")
# Prior
parser.add_argument('-p', '--prior', type=str, metavar='', default='IID', required=False,
                    help="{{'IID', }}, default='IID' \nPrior distribution of the weights. Used for HMC, NUTS, VI, laplace.")
# Likelihood
parser.add_argument('-ll', '--likelihood', type=str, metavar='', default='Homoskedastic Gaussian', required=False,
                    help="{{'Homoskedastic Gaussian', }}, default='Homoskedastic Gaussian' \nLikelihood function. Used for HMC, NUTS, VI.")
# Likelihood scale
parser.add_argument('--likelihood_scale', type=float, metavar='', default=1e-2, required=False,
                    help="default=1e-2 \nScale for Homoskedastic Gaussian likelihood.")
# Kernel
parser.add_argument('-k', '--kernel', type=str, metavar='', default='NUTS', required=False,
                    help="{{'NUTS', 'HMC'}}, default='NUTS' \nSampling algorithm to use when doing MCMC sampling.")
# Number of MCMC samples
parser.add_argument('-s', '--n_samples', type=int, metavar='', default=10000, required=False,
                    help="default=10000 \nNumber of samples to generate from HMC or NUTS.")
# Warmup
parser.add_argument('--n_warmup', type=int, metavar='', default=5000, required=False,
                    help="{{-1, int(0, 10000)}}, default=-1 \nNumber of burn-in samples to discard for HMC or NUTS. n_warmup=-1 automatically discards the first 30% of the total samples.")
# Posterior samples
parser.add_argument('-ps', '--posterior_samples', type=int, metavar='', default=300, required=False,
                    help="default=300 \nNumber of posterior samples to use for evaluation")
# Guide
parser.add_argument('--guide', type=str, metavar='', default='mean field', required=False,
                    help="{{'mean field', }}, default='mean field' \nApproximation distribution for VI.")
# Guide scale
parser.add_argument('--guide_scale', type=float, metavar='', default=1e-2, required=False,
                    help="default=1e-2, \nScale for approximation distribution in  MFVI.")
# Subset of weights for Laplace approximation
#parser.add_argument('--subset_of_weights', type=str, metavar='', default='all', required=False,
#                    help="{{'all', 'last_layer', 'subnetwork'}}, default='all'.\nSubset of network weights to consider for inference.")
# Rounding
parser.add_argument('--rounded', type=int, metavar='', default=10, required=False,
                    help="default=10 \nNumber of rounding digits for printed results.")

# Parse args
args = parser.parse_args()


# This code snippet manually places the working dir at the project's root directory.
if args.loc == 'local':
    path = 'C:/Users/Administrator/Documents/Alex/Master/Internship/github/InternshipMatakos/'
    os.chdir(path)
elif args.loc == 'cluster':
    path = '/scratch/work/matakoa2/github/InternshipMatakos/'
    os.chdir(path)
else:
    raise ValueError("Please specify either --loc='local' for local use or --loc='cluster' for use on the cluster'.")

    
# Set seed
args.seed = npr.randint(10000) if args.seed == 'random' else seeds[args.dataset]
npr.seed(args.seed)
torch.manual_seed(args.seed)

# Set device
args.device = torch.device(args.device)
device = args.device



# Main script
def main(args):

    # Prepare and load data
    datapath = data_path[args.dataset]
    n_data, n_features, n_classes, train_loader, test_loader, OOD_loader = data_preprocessing.prepare_data(datapath, args)

    # Define loss function
    if n_classes == 1:    # regression
        loss_function = nn.MSELoss()
    else:                 # classification
        loss_function = nn.NLLLoss()
        
        
    
    ########################################################### MCMC #####################################################
    if args.inference == 'MCMC':
        # Network
        net = MLP(n_features, n_classes, args.network, args.activation).to(device)
        
        # Prior
        if args.prior == 'IID':
            prior = tyxe.priors.IIDPrior(dist.Normal(
                torch.tensor(0, device=device, dtype=torch.float),
                torch.tensor(1, device=device, dtype=torch.float)))
            
            prior_name = ", IID prior"
        else:
            raise("This type of prior is not implemented yet.")
            
        # Likelihood
        if args.likelihood == 'Homoskedastic Gaussian':
            likelihood = tyxe.likelihoods.HomoskedasticGaussian(n_data, scale=args.likelihood_scale)
            likelihood_name = ", Homoskedastic Gaussian likelihood"
        else:
            raise("This type of likelihood is not implemented yet.")
            
        # Sampler
        if args.n_warmup == -1:
            args.n_warmup = int(0.3 * args.n_samples)   # default setting discards 30% of the samples
        if args.kernel == 'NUTS':
            kernel = partial(pyro.infer.mcmc.NUTS, adapt_step_size=True)
        elif args.kernel == 'HMC':
            kernel = partial(pyro.infer.mcmc.HMC, step_size=1e-3, num_steps=20, target_accept_prob=0.7)
            
        # Combine everything to form the BNN
        bnn = tyxe.bnn.MCMC_BNN(net, prior, likelihood, kernel)
        bnn.name = "MCMC (" + args.kernel +")" + likelihood_name + prior_name
    
        # Start the sampler
        if args.n_warmup == -1:
            args.n_warmup = int(0.3 * args.n_samples)   # default setting discards 30% of the samples
        pyro.clear_param_store()
        bnn.fit(train_loader, args.n_samples, warmup_steps=args.n_warmup)

        # Predict
        train_evaluation = evaluate_regr(bnn, train_loader, loss_function, posterior_samples=args.posterior_samples, 
                                         rounded=args.rounded, dataset=args.dataset + ", Train data", device=device)
        train_evaluation['Neural network'] = args.network
        
        test_evaluation = evaluate_regr(bnn, test_loader, loss_function, posterior_samples=args.posterior_samples,
                                        rounded=args.rounded, dataset=args.dataset + ", Test data", device=device)
        test_evaluation['Neural network'] = args.network
        
        OOD_evaluation = evaluate_regr(bnn, OOD_loader, loss_function, posterior_samples=args.posterior_samples,
                                       rounded=args.rounded, dataset=args.dataset + ", OOD data", device=device)
        OOD_evaluation['Neural network'] = args.network
        
        # Print results
        print_dict(train_evaluation)
        print_dict(test_evaluation)
        print_dict(OOD_evaluation)

        
        
        
        
    ########################################################### VI #######################################################
    
    elif args.inference == 'VI':
        # Network
        net = MLP(n_features, n_classes, args.network, args.activation).to(device)
        
        # Prior
        if args.prior == 'IID':
            prior = tyxe.priors.IIDPrior(dist.Normal(
                torch.tensor(0, device=device, dtype=torch.float),
                torch.tensor(1, device=device, dtype=torch.float)))
            
            prior_name = ", IID prior"
        else:
            raise("This type of prior is not implemented yet.")
            
        # Likelihood
        if args.likelihood == 'Homoskedastic Gaussian':
            likelihood = tyxe.likelihoods.HomoskedasticGaussian(n_data, scale=args.likelihood_scale)
            likelihood_name = ", Homoskedastic Gaussian likelihood"
        else:
            raise("This type of likelihood is not implemented yet.")
            
        # Approximation distribution
        if args.guide == 'mean field':
            guide_builder = partial(tyxe.guides.AutoNormal, init_scale=args.guide_scale)
            guide_name = "Mean field"
        else:
            raise("This type of VI is not implemented yet.")
            
        bnn = tyxe.VariationalBNN(net, prior, likelihood, guide_builder)
        bnn.name = guide_name + " Variational BNN" + likelihood_name + prior_name
        
        # Fit
        pyro.clear_param_store()
        optim = pyro.optim.Adam({"lr": args.learning_rate})
        elbos = []    
        def callback(bnn, i, e):
            elbos.append(e)
        bnn.fit(train_loader, optim, args.n_epochs, callback, device=device)
        
        # Predict
        train_evaluation = evaluate_regr(bnn, train_loader, loss_function, posterior_samples=args.posterior_samples, 
                                         rounded=args.rounded, dataset=args.dataset + ", Train data", device=device)
        train_evaluation['Neural network'] = args.network
        
        test_evaluation = evaluate_regr(bnn, test_loader, loss_function, posterior_samples=args.posterior_samples,
                                        rounded=args.rounded, dataset=args.dataset + ", Test data", device=device)
        test_evaluation['Neural network'] = args.network
        
        OOD_evaluation = evaluate_regr(bnn, OOD_loader, loss_function, posterior_samples=args.posterior_samples,
                                       rounded=args.rounded, dataset=args.dataset + ", OOD data", device=device)
        OOD_evaluation['Neural network'] = args.network
        
        # Print results
        print_dict(train_evaluation)
        print_dict(test_evaluation)
        print_dict(OOD_evaluation)
        


        
        
    ######################################################## Laplace ####################################################        
    
    elif args.inference == 'laplace':
        # Network
        net = MLP(n_features, n_classes, args.network, args.activation).to(device)
        
        # Laplace
        problem_type = 'regression' if n_classes == 1 else 'classification'
        LA = Laplace(net, problem_type, subset_of_weights='all')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    