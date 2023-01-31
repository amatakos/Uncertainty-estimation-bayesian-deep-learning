import numpy as np
import torch
import torch.nn as nn
import pyro
import tyxe

def count_parameters(model):
    """
    Count model parameters for standard NN.
    """
    return sum(p.numel() for p in model.parameters())


def posterior_sd(bnn, likelihood="Homoskedastic Gaussian"):
    """
    Find common shared posterior sd for Homoskedastic Gaussian likelihood.
    """

    posterior = bnn._mcmc.get_samples()

    if likelihood == "Homoskedastic Gaussian":
        samples = torch.empty(0, device='cuda')
        for key in posterior:
            samples = torch.cat((samples, posterior[key].flatten()))

        return samples.std()
    else:
        raise("This type of likelihood is not impleneted yet.")
        
        
def print_dict(dictionary):
    for key, value in dictionary.items():
        print(key + str(":"), value)    
        
    pass


def MLP_to_torch_nn_Sequential(mlp):
    """
    Turn an object from the MLP class (custom defined class) to a 
    nn.Sequential object.
    
    NOTE: This code is hacky and ugly, it only exists to fix some laplace
    incompatibilities.
    """
    
    if 'relu' in str(mlp.act):
        activation = nn.ReLU
    elif 'tanh' in str(mlp.act):
        activation = nn.Tanh
    else:
        activation = nn.Sigmoid
    
    if mlp.batch_norm:
        if len(mlp.hidden_layers) == 1:
            la_model = nn.Sequential(mlp.layer_0, activation(), mlp.layer_1, 
                                     mlp.layer_2)
        elif len(mlp.hidden_layers) == 2:
            la_model = nn.Sequential(mlp.layer_0, activation(), mlp.layer_1, 
                                     mlp.layer_2, activation(), mlp.layer_3,
                                     mlp.layer_4)
        elif len(mlp.hidden_layers) == 3:
            la_model = nn.Sequential(mlp.layer_0, activation(), mlp.layer_1, 
                                     mlp.layer_2, activation(), mlp.layer_3,
                                     mlp.layer_4, activation(), mlp.layer_5,
                                     mlp.layer_6)
        else:
            raise NotImplementedError("Laplace approximation supports NN of up to 3 layers deep.")
    
    else:    
        if len(mlp.hidden_layers) == 1:
            la_model = nn.Sequential(mlp.layer_0,  activation(), mlp.layer_1)

        elif len(mlp.hidden_layers) == 2:
            la_model = nn.Sequential(mlp.layer_0,  activation(), mlp.layer_1,
                                     activation(), mlp.layer_2)

        elif len(mlp.hidden_layers) == 3:
            la_model = nn.Sequential(mlp.layer_0,  activation(), mlp.layer_1,
                                     activation(), mlp.layer_2,
                                     activation(), mlp.layer_3)
        else:
            raise NotImplementedError("Laplace approximation supports NN of up to 3 layers deep.")

            
    # Load the trained weights
    path = 'temp_laplace_model.pt'
    torch.save(mlp.state_dict(), path)
    la_model.load_state_dict(torch.load(path), strict=False)
    
    # Final touch
    setattr(la_model, 'name', 'laplace')
    setattr(la_model, 'hidden_layers', mlp.hidden_layers)
    
    return la_model

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        