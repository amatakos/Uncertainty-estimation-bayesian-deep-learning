import numpy as np
import matplotlib.pyplot as plt


def plot_train(loss=None, acc=None, get_axes=False):
    """
    Current implementation compatible with:
        ONLY fit_class (code written to plot two plots side by side,
        namely the loss and dev accuracy)
        
    Plot epoch-wise avg loss plot and acc 
    
    Arguments
    ---------
    loss: list of floats
        epoch-wise avg loss value on during training
    acc: list of floats
        epoch-wise acc during training
    get_axs: bool
        whether to show plot or return axes and fig
        
    Returns
    -------
    None or axs
    None: if get_axes is False
    (fig, axs): tuple, if get_axes is True
        fig: plt.figure
        axs: list of axes
        
    """
    axs=[]
    
    fig = plt.figure(figsize=(17, 6))
    
    if loss:
        ax1 = fig.add_subplot(121)
        ax1.plot(loss)
        ax1.grid()
        ax1.set_xlabel("Epoch", fontsize=20)
        ax1.set_ylabel("Loss", fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.set_title("Average loss during training", fontsize=20)
        
        if get_axes:
            axs.append(ax1)

    if acc:
        ax2 = fig.add_subplot(122)
        ax2.plot(acc)
        ax2.grid()
        ax2.set_xlabel("Epoch", fontsize=20)
        ax2.set_ylabel("Accuracy", fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_title("Accuracy during training", fontsize=18)
        
        if get_axes:
            axs.append(ax2)
            
    if len(axs)==0:
        plt.show()
        pass
    else:
        return fig, axs