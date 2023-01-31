import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.utils import data_loaders
from src.utils.plotting import plot_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# This module contains fitting and testing routines for regression
# and classification models separately. Only for non-Bayesian NNs.


def fit_regr(model, loss_func, optimizer, loader, n_epochs, verbose=0,
             early_stopping=None, return_losses=False):
    """
    Train the model on train data.
    Used for regression model, standard NN.

    Returns
    -------

    """
    # check if early stopping using dev set is enabled and create dev set if it is
    if early_stopping == 'dev':
        loader, dev_loader = data_loaders.create_dev_set(loader, dev_size=0.2)

    ########################## EARLY STOPPING / plotting related stuff ######################
    losses = []                                      # for plotting loss curve
    loss_queue = [torch.inf for _ in range(50)]      # for early stopping
    dev_loss_queue = [torch.inf for _ in range(50)]  # for early stopping
    #########################################################################################

    for epoch in range(n_epochs):
        loss_epoch = 0
        total = 0
        for batch_num, (data, target) in enumerate(loader):

            model.zero_grad()
            output = model(data)
            loss = loss_func(output, target)

            # Optimize
            loss.backward()
            optimizer.step()

            total += len(target)
            train_loss = loss.item()
            loss_epoch += train_loss

            if verbose > 2:
                print('Training: Epoch %d - Batch %d/%d: MSE Loss: %.8f' % \
                      (epoch + 1, batch_num + 1, len(train_loader), train_loss / (batch_num + 1)))

        if verbose > 1:
                print('Training: Epoch %d: Average MSE Loss: %.8f' % \
                      (epoch + 1, loss_epoch / total))

        losses.append(loss_epoch / total)

        if early_stopping in ['train', True]:
            with torch.no_grad():
                # early stopping checking on the train data
                # stops the training if total loss hasn't decreased on average in 50 epochs

                loss_queue.pop(0)
                loss_queue.append(loss_epoch)

                avg_previous_loss = torch.tensor(loss_queue).mean().item()
                if avg_previous_loss <= loss_epoch:
                    if verbose > 0:
                        print("Loss has stopped decreasing after epoch %d (average of %d epochs).\nSTOPPING EARLY." \
                              %(epoch+1, len(loss_queue)))
                    break

        elif early_stopping == 'dev':
            with torch.no_grad():
                # early stopping checking on dev set
                dev_model = copy.deepcopy(model)

                dev_loss = test_regr(dev_model, loss_func, dev_loader, verbose=0, return_loss=True)

                dev_avg_previous_loss = torch.tensor(dev_loss_queue).mean().item()
                if dev_avg_previous_loss <= dev_loss:
                    if verbose > 0:
                        print("Loss on dev set has stopped decreasing after epoch %d (average of %d epochs).\nSTOPPING EARLY." \
                              %(epoch+1, len(dev_loss_queue)))
                        
                    break
                
                dev_loss_queue.pop(0)
                dev_loss_queue.append(dev_loss)

    if return_losses:
        return losses

    return losses[-1]



def test_regr(model, loss_func, loader, verbose=0, return_loss=False):
    """
    Calculate loss for test data. Used for non-bayesian regression NN.
    """

    total = 0
    loss_total = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(loader):

            output = model(data)
            loss = loss_func(output, target)
            test_loss = loss.item()
            loss_total += test_loss

            total += len(target)
            if verbose == 2:
                print('Evaluating: Batch %d/%d: Loss: %.8f' %
                      (batch_num + 1, len(loader), test_loss / (batch_num + 1)))

        avg_loss = loss_total / total

        if verbose > 0:
            print(f"Average Test MSE Loss: {avg_loss:.8f}")

        if return_loss:
            return avg_loss

    pass


def fit_class(model, loss_func, optimizer, loader, n_epochs, verbose=0, early_stopping=None,
              return_losses=False, return_acc=False):
    """
    Train the model on train data.
    Used for classification model, standard NN.

    Returns
    -------

    """
    # check if early stopping using dev set is enabled and create dev set if it is
    if early_stopping == 'dev':
        loader, dev_loader = data_loaders.create_dev_set(loader, dev_size=0.2)

    ########################## EARLY STOPPING / plotting related stuff ######################
    accuracy_over_epochs = []
    dev_accuracy_over_epochs = []
    losses = []                                      # for plotting loss curve
    loss_queue = [torch.inf for _ in range(50)]      # for early stopping
    dev_acc_queue = [torch.zeros(1) for _ in range(50)]  # for early stopping
    #########################################################################################

    for epoch in range(n_epochs):
        loss_epoch = 0
        total = 0
        correct_total = 0
        for batch_num, (data, target) in enumerate(loader):

            model.zero_grad()
            output = model(data)
            loss = loss_func(output, target)

            # Optimize
            loss.backward()
            optimizer.step()

            total += len(target)
            train_loss = loss.item()
            loss_epoch += train_loss

            predicted = torch.argmax(output, axis=1)
            train_correct = sum(predicted == target)
            correct_total += train_correct

            if verbose > 2:
                print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                      (epoch + 1, batch_num + 1, len(loader), train_loss,
                       100. * train_correct / len(target), train_correct, len(target)))

        if verbose > 1:
            print('Training: Epoch %d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                  (epoch + 1, loss_epoch / total,
                   100. * correct_total / total, correct_total, total))

        accuracy_over_epochs.append(correct_total / total)
        losses.append(loss_epoch / total)

        if early_stopping in ['train', True]:
            with torch.no_grad():
                # early stopping checking on the train data
                # stops the training if total loss hasn't decreased on average in 50 epochs

                loss_queue.pop(0)
                loss_queue.append(loss_epoch)

                avg_previous_loss = torch.tensor(loss_queue).mean().item()
                if avg_previous_loss <= loss_epoch:
                    if verbose > 0:
                        print("Loss has stopped decreasing after epoch %d (average of %d epochs).\nSTOPPING EARLY." \
                              %(epoch+1, len(loss_queue)))
                    break

        elif early_stopping == 'dev':
            with torch.no_grad():
                # early stopping checking on dev set
                # create dev set from the train loader
                dev_model = copy.deepcopy(model)

                dev_acc = test_class(dev_model, loss_func, dev_loader, verbose=0, return_acc=True)
                dev_accuracy_over_epochs.append(dev_acc)

                dev_acc_queue.pop(0)
                dev_acc_queue.append(dev_acc)

                avg_previous_acc = torch.tensor(dev_acc_queue).mean().item()
                if dev_acc < avg_previous_acc:
                    if verbose > 0:
                        print("Accuracy on dev set (%s) has stopped increasing after epoch %d (average of %d epochs).\nSTOPPING EARLY." \
                              %(str(np.round(100*dev_acc, 1))[:4]+'%', epoch+1, len(dev_acc_queue)))
                    break

    accuracy_over_epochs = [acc.item() for acc in accuracy_over_epochs] # convert to floats

    if return_losses:
        if return_acc:

            if early_stopping == 'dev':
                return losses, dev_acc_queue
            else:
                return losses, accuracy_over_epochs

        else:
            return losses

    else:
        if early_stopping == 'dev':
            return dev_acc_queue
        else:
            return accuracy_over_epochs



def test_class(model, loss_func, loader, verbose=1, rounded=16,
               return_loss=False, return_acc=False):
    """
    Test accuracy of model on test data.
    Used for classification model, standard NN.

    Returns
    -------
    None,
        default behavior.
    loss_total: torch.float,
        total loss, optional.
    acc: torch float,
        between 0 and 1, optional.
    """

    total = 0
    total_correct = 0
    loss_total = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(loader):
            output = model(data)
            loss = loss_func(output, target)
            test_loss = loss.item()
            loss_total += test_loss

            predicted = torch.argmax(output, axis=1)
            correct = sum(predicted == target)
            total_correct += correct

            total += len(target)
            if verbose == 2:
                print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
                      (batch_num + 1, len(loader), test_loss / (batch_num + 1),
                       100. * correct / count, correct, count))

        acc = (torch.true_divide(total_correct, total)* 100).cpu().detach().numpy()
        if verbose > 0:
            print(f"Accuracy: {total_correct}/{total} = {np.round(acc, rounded)}%")

    acc = np.round(acc, rounded) / 100
    loss_total = loss_total / total

    if return_loss and return_acc:
        return loss_total, acc
    elif return_loss:
        return loss_total
    elif return_acc:
        return acc
    else:
        pass


def fit_ensemble(model, loss_func, lr=0.001, M=5, plot=False, verbose=0,
                 return_losses=True, return_acc=True, **kwargs):
    """
    Similar to fit_regr or fit_class. Trains M networks and combines them to output
    the average prediction. Automatically determines regression or classification task.

    Arguments
    ---------
    model: nn.Module object,
        the network to be trained
    loss_function: torch.nn function
        the loss function
    optimizer: torch.optim object,
        the optimizer
    M: int,
        number of models to use for the ensemble
    plot_loss: bool
        plot loss curves for each model
    **kwargs:
        inputs to fit_regr

    Returns
    -------
    ensemble: list
        list containing M nn.Module models
    """

    # Determine if it is classification or regression
    task = 'classification' if model.n_classes > 1 else 'regression'

    # Create ensemble
    ensemble = [copy.deepcopy(model) for m in range(M)]
    ensemble_optimizers = [torch.optim.Adam(ensemble[m].parameters(), lr=lr) for m in range(M)]


    if plot:
        if task == 'classification':
            fig, axs = plot_train([None], [None], get_axes=True) # Initialize empty axes
        elif task == 'regression':
            fig, axs = plot_train([None], get_axes=True) # Initialize empty axes


    # Train
    for m in range(M):
        # fit can contain multiple things, check fit_class or fit_regr documentation
        if verbose > 0:
            print("Training network no.", m+1)

        if task == 'classification':
            fit = fit_class(ensemble[m], loss_func, ensemble_optimizers[m], verbose=verbose,
                            return_losses=True, return_acc=True, **kwargs)

            if plot:
                for ax, item in zip(axs, fit):
                    ax.plot(item)

        elif task == 'regression':
            fit = fit_regr(ensemble[m], loss_func, ensemble_optimizers[m], verbose=verbose,
                            return_losses=True, **kwargs)

            if plot:
                axs[0].plot(fit)

    # Show plots
    if plot:
        plt.show()

    return ensemble


def test_class_ensemble(ensemble, loss_func, loader, verbose=1, rounded=16,
        return_loss=False, return_acc=False):
    """
    Test accuracy of ensemble on test data.
    Used for classification.

    Returns
    -------
    None,
        default behavior.
    loss_total: torch.float,
        total loss, optional.
    acc: torch float,
        between 0 and 1, optional.
    """

    n_classes = ensemble[0].n_classes
    total = 0
    total_correct = 0
    loss_total = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(loader):
            probs = [torch.zeros(len(data), len(ensemble), device=ensemble[0].device) for _ in range(n_classes)]

            for m, model in enumerate(ensemble):
                # Output of network is log(p) = log(softmax(logits)) since we are using nn.LogSoftmax
                log_p = model(data)
                p = log_p.exp()

                for j in range(p.shape[1]):
                    probs[j][:, m] = p[:, j]

            for i in range(len(probs)):         # probs is a list of matrices
                probs[i] = probs[i].mean(axis=1)

            # Convert to matrix
            probs = torch.stack(probs, axis=1)

            # Calculate loss and acc
            loss = loss_func(probs, target)
            test_loss = loss.item()
            loss_total += test_loss

            predicted = torch.argmax(probs, axis=1)
            correct = sum(predicted == target)
            total_correct += correct

            total += len(target)
            if verbose == 2:
                print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
                      (batch_num + 1, len(loader), test_loss / (batch_num + 1),
                       100. * correct / count, correct, count))

        acc = (torch.true_divide(total_correct, total)* 100).cpu().detach().numpy()
        if verbose > 0:
            print(f"Accuracy: {total_correct}/{total} = {np.round(acc, rounded)}%")

    acc = np.round(acc, rounded) / 100
    loss_total = loss_total / total

    if return_loss and return_acc:
        return loss_total, acc
    elif return_loss:
        return loss_total
    elif return_acc:
        return acc
    else:
        pass


def test_regr_ensemble(ensemble, loss_func, loader, verbose=0, return_loss=False):
    total = 0
    loss_total = 0
    for data, target in loader:

        # Calculate average ensemble ouptput
        output = torch.empty((len(data), len(ensemble)), device=ensemble[0].device)
        for m, model in enumerate(ensemble):
            output[:, m] = model(data).flatten()

        # Calculate loss
        loss = loss_func(output.mean(axis=1).view(-1, 1), target.view(-1, 1))
        test_loss = loss.item()
        loss_total += test_loss

        total += len(target)

    avg_loss = loss_total / total

    if verbose > 0:
        print(f"Average Test MSE Loss: {avg_loss}")

    if return_loss:
        return avg_loss

    pass


def fit_gmlp(model, optimizer, loader, n_epochs=100, verbose=0,
             early_stopping=None, return_losses=False):
    """
    Train a GaussianMLP.

    A GaussianMLP is a NN which outputs a mean and a variance.

    """

    # check if early stopping using dev set is enabled and create dev set if it is
    if early_stopping == 'dev':
        loader, dev_loader = data_loaders.create_dev_set(loader, dev_size=0.2)


    losses = []
    loss_queue = [torch.inf for _ in range(50)] # for early stopping

    for epoch in range(n_epochs):
        loss_epoch = 0
        total = 0

        for data, target in loader:

            model.zero_grad()
            mu, var = model(data)
            mu, var = mu.flatten(), var.flatten()

            # Negative log-likelihood loss
            loss = (torch.log(var) + ((target - mu).pow(2)) / var).sum()     # constants don't matter for SGD

            # Optimize
            loss.backward()
            optimizer.step()

            total += len(target)
            train_loss = loss.item()
            loss_epoch += train_loss

        losses.append(loss_epoch / total)


        if early_stopping in ['train', True]:
            with torch.no_grad():
                # early stopping checking on the train data
                # stops the training if total loss hasn't decreased on average in 50 epochs

                loss_queue.pop(0)
                loss_queue.append(loss_epoch)

                avg_previous_loss = torch.tensor(loss_queue).mean().item()
                if avg_previous_loss <= loss_epoch:
                    if verbose > 0:
                        print("Loss has stopped decreasing after epoch %d (average of %d epochs).\nSTOPPING EARLY." \
                              %(epoch+1, len(loss_queue)))
                    break

        elif early_stopping == 'dev':
            with torch.no_grad():
                # early stopping checking on dev set
                dev_model = copy.deepcopy(model)

                dev_loss = test_gmm(dev_model, dev_loader, return_loss=True)

                loss_queue.pop(0)
                loss_queue.append(dev_loss)

                dev_avg_previous_loss = torch.tensor(loss_queue).mean().item()
                if dev_avg_previous_loss <= dev_loss:
                    if verbose > 0:
                        print("Loss on dev set has stopped increasing after epoch %d (average of %d epochs).\nSTOPPING EARLY." \
                              %(epoch+1, len(loss_queue)))
                    break


        if verbose > 1:
            print('Training: Epoch %d: Average NLL Loss: %.8f' % \
                  (epoch + 1, loss_epoch / total))

    if return_losses:
        return losses

    return losses[-1]


def fit_gmm(gmm, optimizers, loader, n_epochs, verbose=0, return_losses=False,
           plot_loss=False, early_stopping=False):
    """
    Train a Gaussian Mixture MLP.

    A Gaussian Mixture MLP is a Gaussian approximation of a Deep Ensemble
    consisting of Gaussian MLPs.

    """

    if plot_loss:
        fig = plt.figure(figsize=(9, 6))
        return_losses=True

    gmm_losses = []
    total = 0
    for m in range(gmm.n_models):
        if verbose > 0:
            print("Training network no.", m+1)
        model = getattr(gmm, 'model_'+str(m))
        train_losses = fit_gmlp(model, optimizers[m], loader, n_epochs=n_epochs, verbose=verbose-1,
                             early_stopping=early_stopping, return_losses=return_losses)
        gmm_losses.append(train_losses)

        if plot_loss:
            plt.plot(train_losses)

    if plot_loss:
        plt.grid()
        plt.xlabel("Epoch", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.title("Average NLL loss", fontsize=20)
        plt.show()

    if return_losses:
        return gmm_losses

    pass


def test_gmm(model, loader, verbose=0, return_loss=False, return_total_loss=False):
    """
    Calculate MSE loss for a Gaussian MLP or a Gaussian Mixture MLP.

    """

    total = 0
    loss_total = 0
    with torch.no_grad():
        for data, target in loader:

            mu, var = model(data)
            loss = ((mu.flatten() - target)**2).sum()

            test_loss = loss.item()
            loss_total += test_loss

            total += len(target)

        avg_loss = loss_total / total

        if verbose > 0:
            print(f"Average Test MSE Loss: {avg_loss:.8f}")

        if return_loss:
            if return_total_loss:
                return avg_loss, loss_total
            else:
                return avg_loss

    pass








































