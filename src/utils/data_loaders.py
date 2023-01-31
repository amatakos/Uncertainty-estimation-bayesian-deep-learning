import torch

def create_torch_dataset(X, y, to_LongTensor=True):
    """
    Takes two numpy arrays X, y and converts them to a TensorDataset
    compatible with torch data loaders.
    
    Arguments
    ---------
    X:  (n, k) np.array
        data
    y:  (n,) np.array
        labels
    to_LongTensor: bool,
        convert targets to torch.LongTensor. Needed in some cases
        
    Returns
    -------
    dataset: TensorDataset
    """
    
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    if to_LongTensor:
        y = y.type(torch.LongTensor)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    
    return dataset


def separate_torch_dataset(dataset, device="cpu"):
    """Separates a torch.util.data.Dataset into data (X) and targets (y).

    Arguments
    ---------
    dataset: torch.util.data.Dataset object
    device: torch.device, 'cpu' or 'cuda'.

    Returns
    -------
    X: torch.tensor, size (n,k),
           where n=len(dataset), k=number of features
    y: torch.tensor, size (n,)
    """
    X, y = [], []
    for img, label in dataset:
        X.append(img)
        y.append(label)
    X, y = torch.stack(X).to(device), torch.tensor(y).to(device)

    return X, y


def create_subset(dataset, subset_size, complement=False):
    """
    Creates torch subset of size=subset_size of the given dataset.

    Arguments
    ---------
    dataset:  torch.utils.data.Dataset object
    subset_size: int, desired size of subset
    complement: bool, default False
        whether to return the complement of the selected subset

    Returns
    -------
    subset: torch.utils.data.Subset object, default behavior
            subset of the given dataset.
    compl: torch.utils.data.Subset object, 
        the complement of the selected subset.
    """

    perm = torch.randperm(len(dataset))
    idx = perm[:subset_size]
    subset = torch.utils.data.Subset(dataset, idx)

    if complement:
        compl_idx = perm[subset_size:]
        compl = torch.utils.data.Subset(dataset, compl_idx)
        
        return subset, compl
    
    return subset


def create_loader(dataset, batch_size=100, device='cpu', num_workers=0):
    """
    Creates a dataset's loader. Loads data on cuda device if needed.

    Arguments
    ---------
    dataset:  torch.utils.data.Dataset object
    batch_size: int, minibatch size.
    device: torch.device, 'cpu' or 'cuda'

    Returns
    -------
    loader: torch.utils.data.DataLoader object,
            loader of the given dataset.
    """

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         # shuffle=True,
                                         shuffle=False,
                                         num_workers=num_workers)

    if str(device) == 'cuda':
        loader = loader_to_cuda(loader)

    return loader


def loader_to_cuda(loader):
    """
    Place a loader's data on GPU. PyTorch does not support this so it has
    to be done manually. First iterate through loader, take the data,
    pass it to cuda device, then create a new dataset and finally wrap it
    with a loader again.

    Arguments
    ---------
    loader: torch.util.data.DataLoader object

    Returns
    -------
    loader: torch.util.data.DataLoader object
    """
    device = 'cuda'


    # Initialize with first batch
    data, target = next(iter(loader))
    data, target = data.to('cuda'), target.to('cuda')

    for i, (x, y) in enumerate(loader):
        if i==0:        # skip the first batch
            continue

        x, y  = x.to(device), y.to(device)
        data = torch.cat((data, x))
        target = torch.cat((target, y))

    dataset = torch.utils.data.TensorDataset(data, target)
    loader = create_loader(dataset, loader.batch_size)

    return loader


def create_dev_set(loader, dev_size=0.2):
    """
    TODO: Add if else statements to support TensorDataset input.
    
    Create dev set from a torch.DataLoader (usually train loader).
    Implicitly inherits torch.device.
    
    Returns
    -------
    dev_loader: torch.DataLoader containing development data set.
    """

    data, target = next(iter(loader))
    
    for i, (x,y) in enumerate(loader):
        if i==0: # skip first batch
            continue
         
        data = torch.cat((data, x))
        target = torch.cat((target, y))
        
    dataset = torch.utils.data.TensorDataset(data, target)
    
    subset_size = int(len(data) * dev_size)
    dev, train = create_subset(dataset, subset_size, complement=True)
    
    train_loader = create_loader(train, loader.batch_size)
    dev_loader = create_loader(dev, loader.batch_size)
    
    return train_loader, dev_loader
    
    
    
    
    
    
    
    
    
    
    
    
    
