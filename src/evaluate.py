import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch
from torch.distributions.normal import Normal as norm
import torchmetrics as tm
import pyro
import pyro.distributions as dist
import tyxe
from src.utils import util
from src.train_test import test_gmm


def r2score_bnn(model, loader, num_predictions=1000, device='cpu'):
    """
    Calculates R2-score for a BNN regression model.
    """

    r2score = tm.R2Score().to(device)

    preds = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    with torch.no_grad():
        for data, target in loader:

            pred, _ = bnn.predict(data, num_predictions=10)
            preds = torch.cat((preds, pred))
            targets = torch.cat((targets, target))

    r2 = r2score(preds, targets)

    return r2



def r2score(model, loader, device='cpu'):
    """
    Calculates R2-score for a standard NN regression model.
    """

    r2score = tm.R2Score().to(device)

    preds = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    with torch.no_grad():
        for data, target in loader:

            pred = model(data)
            preds = torch.cat((preds, pred))

            targets = torch.cat((targets, target))

    r2 = r2score(preds, targets)

    return r2


def NLPD(
    model,
    loader,
    posterior_samples,
    likelihood = "Homoskedastic Gaussian",   # TODO: obtain this from model, implement other likelihoods
    device='cpu',
    verbose=0,
    ):
    """
    Calculate mean, sd and total NLPD for regression model.
    Used inside evaluate_regr().
    """

    if likelihood == "Homoskedastic Gaussian":
        sd = model.likelihood.scale
    else:
        raise ValueError("This type of likelihood is not implemented yet.")

    # torch tensor with number of posterior samples
    T = torch.tensor(posterior_samples)

    NLPD = torch.empty(0, device=device)
    with torch.no_grad():
        for data, target in loader:
            # pred is a matrix with dimensions (posterior_samples, batch_size, 1)
            pred = model.predict(data, num_predictions=posterior_samples, aggregate=False)

            # warning: heavily vectorized operations ahead!
            if likelihood == "Homoskedastic Gaussian":
                # we transpose pred matrix and vanish the 3rd dimension (it's just 1)
                # resulting matrix is of dimension (batch_size, posterior_samples)
                
                
                log_p = norm(pred[:, :, -1].T, sd).log_prob(target)
            else:
                raise ValueError("This type of likelihood is not implemented yet.")

            # calculate NLPD
            # log_p is now a matrix, we calculate logsumexp along axis=1
            nlpd = -( torch.logsumexp(log_p, 1) - T.log() )
            NLPD = torch.cat((NLPD, nlpd))

    avg_NLPD = NLPD.mean()
    sd_NLPD = NLPD.std()
    total_NLPD = NLPD.sum()

    if verbose==1:
        print("Average NLPD =", avg_NLPD.cpu().detach().numpy())
        print("SD of NLPD =", sd_NLPD.cpu().detach().numpy())
        print("Total NLPD =", total_NLPD.cpu().detach().numpy())

    return avg_NLPD, sd_NLPD, total_NLPD


def NLPD_laplace(la, loader, posterior_samples, device='cpu'):
    """
    TODO: implement NLPD for classification here.

    Calculates average, sd and total NLPD for a regression BNN with
    Laplace Approximation of the posterior.

    Arguments
    ---------
    la: Laplace model,
    loader: pytorch data loader,
    posterior_samples: int, number of posterior samples to use for prediction,
    device: torch.device

    Returns
    -------
    avg_NLPD: the mean NLPD,
    sd_NLPD: the sd of the NLPD,
    total_NLPD: the total NLPD.

    """


    NLPD = torch.empty(0, device=device)

    if la.likelihood == 'regression':
        with torch.no_grad():

            for data, target in loader:
                f_mu, f_var = la.predictive(data, n_samples=posterior_samples, pred_type='nn', link_approx='mc', )
                pred_sd = (f_var + la.sigma_noise**2).sqrt()
                pred_sd = f_var.sqrt()
                f_mu, pred_sd = f_mu.flatten(), pred_sd.flatten()

                log_p = norm(f_mu, pred_sd).log_prob(target)
                # The previous vectorized operation returns the log probability for each target and
                # for each output f_mu. We select the log_p's of the corresponding target-output pairs.
                log_p = torch.diag(log_p)

                # nlpd = -1 * (log_p - torch.tensor(posterior_samples).log())
                nlpd = -log_p
                NLPD = torch.cat((NLPD, nlpd))

        avg_NLPD = NLPD.mean()
        sd_NLPD = NLPD.std()
        total_NLPD = NLPD.sum()

        return avg_NLPD, sd_NLPD, total_NLPD


def NLPD_standard_MLP_ensemble(ensemble, loader, verbose=0):
    """
    Calculate mean, sd and total NLPD for Standard MLP Ensemble.

    Use a gaussian with mean the outputs of the ensemble's networks
    and variance the observed variance of the outputs.

    """

    device = ensemble[0].device
    NLPD = torch.empty(0, device=device)

    with torch.no_grad():
        for data, target in loader:
            output = torch.empty(len(data), len(ensemble), device=device)
            for m, model in enumerate(ensemble):
                output[:, m] = model(data).flatten()

            f_mu = output.mean(axis=1)
            pred_sd = output.std(axis=1)

            log_p = norm(f_mu, pred_sd).log_prob(target)

            nlpd = -log_p.diag().view(-1, 1)
            NLPD = torch.cat((NLPD, nlpd))

    avg_NLPD = NLPD.mean()
    sd_NLPD = NLPD.std()
    total_NLPD = NLPD.sum()

    if verbose==1:
        print("Average NLPD =", avg_NLPD.cpu().detach().numpy())
        print("SD of NLPD =", sd_NLPD.cpu().detach().numpy())
        print("Total NLPD =", total_NLPD.cpu().detach().numpy())

    return avg_NLPD, sd_NLPD, total_NLPD


def NLPD_gmm(model, loader, verbose=0):
    """
    Calculate mean, sd and total NLPD for Deep Ensemble Gaussian Mixture
    regression model.

    """

    device = model.device
    NLPD = torch.empty(0, device=device)

    with torch.no_grad():
        for data, target in loader:
            f_mu, f_var = model(data)
            # f_mu, f_var = f_mu.flatten(), f_var.flatten()
            pred_sd = f_var.sqrt()         # possibly implement noise here?

            log_p = norm(f_mu, pred_sd).log_prob(target)

            # calculate NLPD
            nlpd = -log_p.diag().view(-1, 1)
            NLPD = torch.cat((NLPD, nlpd))

    avg_NLPD = NLPD.mean()
    sd_NLPD = NLPD.std()
    total_NLPD = NLPD.sum()

    if verbose==1:
        print("Average NLPD =", avg_NLPD.cpu().detach().numpy())
        print("SD of NLPD =", sd_NLPD.cpu().detach().numpy())
        print("Total NLPD =", total_NLPD.cpu().detach().numpy())

    return avg_NLPD, sd_NLPD, total_NLPD


def laplace_regr_loss(
    la,
    loss_func,
    loader,
    posterior_samples,
    verbose=0
    ):
    """
    Used for regression BNNs with Laplace Approximation of the posterior.
    Calculates average and total loss (average over posterior samples).

    Returns
    -------
    loss_total: total loss over the given data
    avg_loss: average loss over the given data
    """

    total = 0
    loss_total = 0
    avg_sd = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(loader):

            m, sd = la.predictive(data, n_samples=posterior_samples, pred_type='glm', link_approx='mc', )

            loss = loss_func(m, target)
            test_loss = loss.item()
            loss_total += test_loss

            total += len(target)

            if verbose == 2:
                print('Evaluating: Batch %d/%d: Loss: %.8f' %
                      (batch_num + 1, len(loader), test_loss / (batch_num + 1)))

        avg_loss = loss_total / total

        if verbose > 0:
            print(f"Average Test MSE Loss: {avg_loss}")

    return avg_loss, loss_total


def calc_loss_homoskedastic_gaussian(
    model,
    loss_func,
    loader,
    num_predictions,
    verbose=0,
    return_avg_sd=False
    ):
    """
    Used for regression BNNs.
    Calculates average and total loss (average over posterior samples).
    Likelihood is homoskedastic gaussian -> calculates average common sd.

    Returns
    -------
    loss_total: total loss over the given data
    avg_loss: average loss over the given data
    avg_sd: average sd over the given data (implies the posterior has shared sd)
    """

    total = 0
    loss_total = 0
    avg_sd = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(loader):

            m, sd = model.predict(data, num_predictions=num_predictions)

            loss = loss_func(m, target)
            test_loss = loss.item()
            loss_total += test_loss

            avg_sd += sd.mean()

            total += len(target)

            if verbose == 2:
                print('Evaluating: Batch %d/%d: Loss: %.8f' %
                      (batch_num + 1, len(loader), test_loss / (batch_num + 1)))

        avg_sd /= (batch_num+1)
        avg_loss = loss_total / total

        if verbose > 0:
            print(f"Average Test MSE Loss: {avg_loss}")

    if return_avg_sd:
        return avg_loss, loss_total, avg_sd
    else:
        return avg_loss, loss_total


def evaluate_class(
    model,
    loader,
    posterior_samples=32,
    device='cpu',
    dataset='',
    num_classes=1,
    rounded=16,
    verbose=False,
    ):

    """
    TODO: Merge with evaluate_regr

    Model evaluation function. Calculates evaluation metrics for the BNN.

    Arguments
    ---------
    model: tyxe._BNN or laplace.Laplace (so far),
                    the trained BNN to be evaluated
    loader: torch.utils.data.DataLoader object,
                    on GPU by default
    posterior_samples: int,
                    number of posterior samples used for prediction
    device: torch.device,
                    'cuda' or 'cpu'
    dataset: str,
                    name of dataset eg. 'MNIST'
    num_classes: int, default=1 (regression)
                    number of classes for classification task
    rounded: int,
                    number of decimal digits to show
    verbose: bool,
                    If True, returns also 'targets' and 'probs'

    Returns
    -------
    Dictionary containing the following key value pairs (averages are over
    posterior samples):

    "Evaluated on": dataset, str
                    name of dataset which was evaluated
    "Posterior samples": posterior_samples, int
                    number of posterior samples used for prediction
    "Average NLPD": avg_nlpd, float
                    average negative log-likelihood
    "Average accuracy": avg_acc, float
                    average accuracy
    "Average AUROC": avg_auroc, float
                    average AUROC (closer to 1 is better)
    "Average ECE": avg_ece, float
                    average Top-class Expected Callibration Error
     "Targets":    targets, np.array of size (len(dataset), )
                    target labels of evaluated data
    "Probabilities": probs, np.array: size (len(dataset), num_classes)
                    probability of each class for each evaluation data
    }

    """

    warnings.filterwarnings(
        "ignore",
        message="Metric `AUROC` will save all targets "
        "and predictions in buffer. For large "
        "datasets this may lead to large memory "
        "footprint.",
    )

    targets = []
    probabilities = []

    accuracy = tm.Accuracy().to(device)
    NLPD = tm.MeanMetric().to(device)
    sd_NLPD = tm.MeanMetric().to(device)
    AUROC = tm.AUROC(num_classes).to(device)
    ECE = tm.CalibrationError().to(device)

    with torch.no_grad():
        N = 0
        for data, target in loader:
            N += len(target)

            data = data.to(device)
            target = target.to(device)

            # HMC or VI
            if 'bnn' in model.name.lower():
                logits = model.predict(data, num_predictions=posterior_samples).detach().cpu()

            # Laplace approximation
            elif 'laplace' in model.name.lower():
                logits = model.predictive(data, n_samples=posterior_samples, pred_type='glm', link_approx='probit')

            # Deep ensemble
            elif 'ensemble' in model.name.lower():
                logits = model.predictive(data).detach().cpu()


            logits = logits.to(device)

            accuracy(logits, target)
            NLPD(F.nll_loss(logits, target))
            sd_NLPD( F.nll_loss(logits, target).pow(2) )
            AUROC(logits, target)
            ECE(logits, target)


            targets.append(target)
            probabilities.append(logits)


    targets = torch.cat(targets).detach().cpu().numpy()
    probabilities = torch.cat(probabilities).detach().cpu().numpy()

    avg_nlpd = NLPD.compute().item()
    sd_nlpd = np.sqrt( sd_NLPD.compute().item() - avg_nlpd ** 2 )    # Var = E[X^2] - E[X]^2

    avg_nlpd       = np.round( avg_nlpd, rounded)
    sd_nlpd        = np.round( sd_nlpd, rounded)
    total_nlpd     = np.round( avg_nlpd * N, rounded)
    avg_acc        = np.round( accuracy.compute().item(),  rounded)
    avg_auroc      = np.round( AUROC.compute().item(),  rounded)
    avg_ece        = np.round( ECE.compute().item(),  rounded)


    metrics = {
        "Inference type": model.name,
        "Evaluated on": dataset,
        "Neural network": model.hidden_layers,
        "Number of posterior samples": posterior_samples,
        "Average NLPD": avg_nlpd,
        "SD of NLPD": sd_nlpd,
        "Total NLPD": total_nlpd,
        "Average accuracy": avg_acc,
        "Average AUROC": avg_auroc,
        "Average ECE": avg_ece,
    }

    if verbose:
        metrics['Targets'] = targets
        metrics['Probabilities'] = probabilities

    return metrics


def evaluate_regr(
    model,
    loader,
    loss_func=nn.MSELoss(),
    posterior_samples=1000,
    dataset='',
    device='cpu',
    rounded=16,
    verbose=False,
    ):

    """
    Model evaluation function. Calculates evaluation metrics for the BNN.

    Arguments
    ---------
    model: tyxe._BNN,
                    the trained BNN to be evaluated
    loader: torch.utils.data.DataLoader object
                    Data loader
    loss_func: torch.nn.functional object
                    loss function used to calculate loss
    posterior_samples: int,
                    number of posterior samples used for prediction
    device: torch.device,
                    'cuda' or 'cpu'
    dataset: str,
                    name of dataset, eg. 'MNIST'
    rounded: int,
                    number of decimal digits to show
    verbose: bool,
                    If True, returns also test targets and test probabilities

    TODO: document avg_nlpd, ..., avg_sd


    Returns
    -------
    Dictionary containing the following key value pairs (averages are over
    posterior samples):

    "Evaluated on": dataset, str
                    name of dataset which was evaluated
    "Posterior samples": posterior_samples, int
                    number of posterior samples used for prediction
    "Average NLL": avg_nll, float
                    average negative log-likelihood
    }

    """
    
    if 'mcmc' in model.name.lower().split() or 'variational' in model.name.lower().split():
        avg_NLPD, sd_NLPD, total_NLPD = NLPD(model, loader, posterior_samples, device=device)
        avg_loss, total_loss = calc_loss_homoskedastic_gaussian(model, loss_func, loader,
                                                                num_predictions=posterior_samples, verbose=0)

    elif 'laplace' in model.name.lower().split():
        avg_NLPD, sd_NLPD, total_NLPD = NLPD_laplace(model, loader, posterior_samples, device=device)
        avg_loss, total_loss = laplace_regr_loss(model, loss_func, loader, posterior_samples=posterior_samples)

    elif 'ensemble' in model.name.lower().split():
        avg_NLPD, sd_NLPD, total_NLPD = NLPD_gmm(model, loader)
        avg_loss, total_loss = test_gmm(model, loader, return_loss=True, return_total_loss=True)

    else:
        raise ValueError("Invalid inference type.")

    avg_nlpd   = np.round( avg_NLPD.cpu().detach().numpy(), rounded)
    sd_nlpd    = np.round( sd_NLPD.cpu().detach().numpy(), rounded)
    total_nlpd = np.round( total_NLPD.cpu().detach().numpy(), rounded)
    avg_loss   = np.round( avg_loss, rounded)
    total_loss = np.round( total_loss, rounded)


    metrics = {
        "Inference type": model.name,
        "Evaluated on": dataset,
        "Neural network": model.hidden_layers,
        "Number of posterior samples": posterior_samples,
        "Average NLPD": avg_nlpd,
        "SD of NLPD": sd_nlpd,
        "Total NLPD": total_nlpd,
        "Average loss": avg_loss,
        "Total loss": total_loss,
        }

    return metrics

























