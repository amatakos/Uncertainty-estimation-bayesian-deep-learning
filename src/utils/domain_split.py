import numpy as np
import numpy.linalg as npl
import numpy.random as npr
from sklearn.preprocessing import StandardScaler


def hyperplane_split(X, OOD_size=0.2, eta=1e-0, eps=1e-3, verbose=0, return_plane=0, seed=3407):
    """
    Splits data into train-test datasets according to
    a shifting hyperplane. First, a random direction is chosen
    and then the corresponding hyperplane is used to separate data into
    two datasets: Domain (dom) and Out Of Domain (OOD).
    The hyperplane is moved towards the direction of the random normal until 
    the desired train-test split is achieved.
    
    Arguments
    ---------
    X:  (N, k) np.array
        unlabeled data
    OOD_size: float
        desired in domain / OOD split ratio
    r:  float
        current OOD to total data ratio
    eta: float
        "learning rate", affects how far the hyperplane moves at each step
    eps: float
        threshold for stopping
    
    Returns
    -------
    dom_idx, OOD_idx: (N,) np.arrays, indeces of domain / OOD data
    """
    np.random.seed(seed)
    
    N = X.shape[0]
    
    # Normalize data
    X = StandardScaler().fit_transform(X)
    
    # Pick random direction and normalize
    # https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_on_the_(n_%E2%88%92_1)-sphere
    n = npr.randn(X.shape[1])
    n /= npl.norm(n)
    if verbose:
        print("Found random direction n =", n)
    
    # Loop to calculate the best c
    c = 0       # initial c
    r = 1       # initial ratio r
    while np.abs(r-OOD_size) > eps:
        # the quantity n^T * X + c tells us where the data belongs based on positive or negative sign (similar to SVM)
        signs = np.sign(np.dot(X, n) + c) 

        # Find the ID and OOD data
        OOD = np.sum(signs <= -1)   # points that lie on the hyperplane are counted as ID data
        r = OOD / N                 # new ratio
            
        # Update hyperplane
        dr = r - OOD_size
        c += eta * dr
    
    if verbose:
        print("Found split with ratio r =", r)
    
    # Separate data according to the hyperplane found  
    dom_idx = signs > -1
    OOD_idx = ~(signs > -1)
    
    if return_plane:
        return dom_idx, OOD_idx, n, c
    
    return dom_idx, OOD_idx




def hypersphere_split(X, OOD_size=0.2, eta=1e-0, eps=1e-3):
    """
    Splits data into train-test datasets according to
    an expanding hypersphere. First, the hypersphere starts 
    out as a small ball around 0 and gradually grows until 
    it encompasses (1-OOD_size) percent of the data. This data is 
    regarded as in domain (dom) data, while the rest is 
    regarded as Out Of Domain (OOD).
    
    Arguments
    ---------
    X:  (N, k) np.array
        unlabeled data
    OOD_size: float
        desired in domain / OOD split ratio
    p:  float
        current sphere radius
    r:  float
        current OOD to total data ratio    
    eta: float
        "learning rate", affects how far the hyperplane moves at each step
    eps: float
        threshold for stopping
    
    Returns
    -------
    dom_idx, OOD_idx: (N,) np.arrays, indeces of domain / OOD data
    """
    
    N = X.shape[0]
    
    # Normalize data
    X = StandardScaler().fit_transform(X)    
    
    # Loop to calculate the best radius p
    p = 0       # initial radius
    r = 1       # initial ratio r
    while np.abs(r-OOD_size) > eps:
        
        # data that are more than a radius away from 0 are outside the hypersphere
        OOD = np.sum(npl.norm(X, axis=1) > p)
        r = OOD / N                 # new ratio
            
        # Update radius
        dr = r - OOD_size
        p += eta * dr
    
    print("Found radius p =", p)
      
    print("Found split with ratio r =", r)
    
    # Get the domain / OOD indeces
    dom_idx = npl.norm(X, axis=1) > p
    
    return dom_idx, ~dom_idx