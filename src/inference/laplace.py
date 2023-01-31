from src import evaluate
from src.evaluate import evaluate_regr
import src.utils
from src.utils import util
from src.utils import data_loaders
from src.utils.domain_split import hyperplane_split
from src.train_test import fit_regr
from src.train_test import test_regr
from src.model import MLP

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from laplace import Laplace, marglik_training
from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN








def laplace_model(mlp, problem_type, subset_of_weights='all', hessian_structure='kron', sigma_noise=1e-2):
    """
    Takes a torch.nn.Module model and turns it into a laplace model by using the Laplace() function 
    from the laplace library. After that it adds two attributes to the model, 'hidden_layers' and 'name'.
    This is done to unify models under evaluate() treatment.
    """
    
    la_model = util.MLP_to_torch_nn_Sequential(mlp)
    LA = Laplace(la_model, problem_type, 
                 subset_of_weights=subset_of_weights, 
                 hessian_structure=hessian_structure, 
                 sigma_noise=sigma_noise if problem_type=='regression' else 1.,
                )
    setattr(LA, 'hidden_layers', la_model.hidden_layers)
    
    if subset_of_weights == 'all':
        name = 'Laplace approximation full network'
    elif subset_of_weights == 'last_layer':
        name = 'Laplace approximation last layer'
    else:
        name = 'Laplace approximation custom network subset'
    setattr(LA, 'name', name)
    
    return LA


def main(mlp, train_loader):
    la_model = util.MLP_to_torch_nn_Sequential(mlp)

    LA = laplace_model(la_model)
    LA.fit(train_loader)
    LA.optimize_prior_precision(method='marglik')
    
    pass
    
if __name__ == '__main__':
    # script replacement here
    main(mlp, train_loader)




