import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Fully connected MLP. Network architecture is automatically determined from
    hidden_layers.
    """
    
    def __init__(self, input_dim, output_dim=1, hidden_layers=[100, 30], 
                 activation='relu', batch_norm=True, problem_type='regression'):
        super(MLP, self).__init__()
        self.name = 'Standard MLP'
        self.n_features = input_dim
        self.n_classes = output_dim
        self.hidden_layers = hidden_layers
        self.n_layers = len(hidden_layers)
        self.net_structure = [input_dim, *hidden_layers, output_dim]
        self.batch_norm = batch_norm
        self.problem_type = problem_type
        
        # Choose activation function
        if activation == 'relu':
            self.act = torch.relu
        elif activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'sigmoid':
            self.act = torch.sigmoid
        elif activation == 'abs':
            self.act = torch.abs
        else:
            assert('Use "relu","tanh" or "sigmoid" as activation.')
            
            
        # Create network
        if self.batch_norm:              # With batch normalization
            i = 0
            j = 0
            while i < ( len(self.net_structure) + 1):
                if i % 2 == 0:
                    setattr(self, 'layer_' + str(i), nn.Linear(self.net_structure[i-j], self.net_structure[i-j+1]))
                    j += 1
                else:
                    setattr(self, 'layer_' + str(i), nn.BatchNorm1d(self.net_structure[j]))
                i += 1  
        
        else:                       # Without batch normalization
            for i in range(self.n_layers + 1):
                setattr(self, 'layer_' + str(i), nn.Linear(self.net_structure[i], self.net_structure[i+1]))
    
    def forward(self, x):
        
        if self.batch_norm:
            for i in range(len(self.net_structure)):
                layer = getattr(self, 'layer_' + str(i))
                if i % 2 == 0:            # batch norm
                    x = self.act(layer(x))
                else:                     # no batch norm
                    x = layer(x)
            
            layer = getattr(self, 'layer_' + str(len(self.net_structure)))
            x = layer(x)
                
        else:
            for i in range(self.n_layers):
                layer = getattr(self, 'layer_' + str(i))
                x = self.act(layer(x))
        
            layer = getattr(self, 'layer_' + str(self.n_layers))
            x = layer(x)
        
        # This is the only difference between regression and classification
        if self.problem_type == 'classification':
            x = nn.LogSoftmax(dim=-1)(x)
            
        return x
    
    @property
    def device(self):
        return next(self.parameters()).device

   
    
class GaussianMLP(MLP):
    """
    Gaussian MLP that outputs mean and variance.
        
    Attributes:
        input_dim (int): number of inputs
        output_dim (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """
    
    def __init__(self, input_dim, output_dim=1, hidden_layers=[30, 30], activation='relu', batch_norm=True):
        super(GaussianMLP, self).__init__(input_dim=input_dim, output_dim=2*output_dim, hidden_layers=hidden_layers, 
                                          activation=activation, batch_norm=batch_norm)
        self.name = 'GaussianMLP'
        self.n_features = input_dim
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        
    def forward(self, x):
        if self.batch_norm:
            for i in range(len(self.net_structure)):
                layer = getattr(self, 'layer_' + str(i))
                if i % 2 == 0:            # batch norm
                    x = self.act(layer(x))
                else:                     # no batch norm
                    x = layer(x)
            
            layer = getattr(self, 'layer_' + str(len(self.net_structure)))
            x = layer(x)
                
        else:
            for i in range(self.n_layers):
                layer = getattr(self, 'layer_' + str(i))
                x = self.act(layer(x))
        
            layer = getattr(self, 'layer_' + str(self.n_layers))
            x = layer(x)
        
        mu, var = torch.split(x, self.output_dim, dim=1)
        var = F.softplus(var) + 1e-6     # ensure positivity
        
        return mu, var
    
    @property
    def device(self):
        return next(self.parameters()).device
    

class GaussianMixtureMLP(nn.Module):
    """ Gaussian mixture MLP that outputs mean and variance.

    Attributes:
        n_models (int): number of models
        input_dim (int): number of inputs
        output_dim (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """
    
    def __init__(self, n_models=5, input_dim=1, output_dim=1, hidden_layers=[30, 30], 
                 activation='relu', batch_norm=True):
        super(GaussianMixtureMLP, self).__init__()
        self.n_models, self.M = n_models, n_models  # for convenience
        self.name = 'Deep Ensemble (Gaussian Mixture of %d MLPs)' % self.n_models
        self.input_dim, self.n_features = input_dim, input_dim
        self.output_dim, self.n_classes = output_dim, output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.batch_norm = batch_norm
        
        for m in range(self.n_models):
            model = GaussianMLP(self.n_features, self.output_dim, self.hidden_layers, self.activation, self.batch_norm)
            setattr(self, 'model_'+str(m), model)
            
    def forward(self, x):
        mu = []
        var = []
        for m in range(self.n_models):
            model = getattr(self, 'model_' + str(m))
            mean, variance = model(x)
            mu.append(mean)
            var.append(variance)
        mus = torch.stack(mu)
        mu = mus.mean(dim=0)
        var = torch.stack(var)
        var = (var + mus.pow(2)).mean(dim=0) - mu.pow(2)
        
        return mu, var

    @property
    def device(self):
        return next(self.parameters()).device
    

class DeepEnsembleClassification(list):
    """ 
    Hacky way of giving some properties (name, device) to the classification Deep Ensemble.
    The reason for this is that I implemented it as a list of NN models. 
    
    Predictive method unifies it with TyXe bnns and laplace.
    
    """
    def __init__(self, ensemble):
        for model in ensemble:
            self.append(model)
            
        self.name = 'Deep Ensemble Classification (' + str(len(self)) + ' models)'
        self.hidden_layers = self[0].hidden_layers
        
    
    def predictive(self, data):
        
        if len(self) == 1:                          # Special case of singleton ensemble (1 MLP)
            log_logits = self[0](data)              # Output of network is log(p) = log(softmax(logits)) since we are using nn.LogSoftmax
            logits = log_logits.exp()
        
        else:
            probs = [torch.zeros(len(data), len(self), device=self[0].device) for _ in range(self[0].n_classes)]
            
            for m, mlp in enumerate(self):
                # Output of network is log(p) = log(softmax(logits)) since we are using nn.LogSoftmax
                log_p = mlp(data)
                p = log_p.exp()
                
                for j in range(p.shape[1]):
                    probs[j][:, m] = p[:, j]
                    
            for i in range(len(probs)):              # probs is a list of matrices (probably could implement it as 3D tensor?)
                probs[i] = probs[i].mean(axis=1)
                
            logits = torch.stack(probs, axis=1)
            
        return logits
            
            
            
    @property
    def device(self):
        return next(self[0].parameters()).device
    

class CNN_MNIST(nn.Module):
    """CNN architecture:
        1st Convolutional layer:
            input channels : 1 (gray)
            output channels : 16
            kernel size : 5
            padding: 2
        2nd Convolutional layer:
            input channels : 16
            output channels : 32
            kernel size : 5
            padding : 2
        Fully connected layer:
            input channels : 32 * 7 * 7
            output channels : 10
            """
    
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=2
                            ),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        )

        self.conv2 = nn.Sequential(
                        nn.Conv2d(16, 32, 5, 1 ,2),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )

        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)      # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)

        return output   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    