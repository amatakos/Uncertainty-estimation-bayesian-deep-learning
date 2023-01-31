# Description

This project was developed during my summer internship in Prof. Arno Solin's Machine Learning Research group. The project's goal was to compare various methods for approximate inference in Bayesian Neural Networks, such as Variatonal Inference methods (ADVI, MFVI), MCMC samplers (HMC, NUTS), Laplace approximation and Ensemble of NNs . This would be thenused to discover principles that dictate how different design choices of BNNs affect the approximation quality of the posterior with respect to the various methods, and assess the efficacy of these methods for uncertainty quantification.

## Installation

Follow the instructions at 'installation instructions.txt'. Requires miniconda and python3.

## Usage

Run 

```python run.py <args>``` 

from root directory. For example, we can create a Bayesian Neural Network with 2 hidden layers (128 and 32 nodes), sigmoid activation function, Laplace approximation for the posterior with 10000 posterior samples, cpu as device and and the 'airfoil' dataset by typing:

```python run.py -d airfoil -net 128 32 -a sigmoid -i laplace -ps 10000 -dv cpu```

(note, that you need to modify the args.location parameter in the run.py script to point to the repo's root directory)

## Results

The above run will yield the following results: 

```
Inference type: Laplace approximation full network
Evaluated on: airfoil, Test data
Neural network: [128, 32]
Number of posterior samples: 10000
Average NLPD: 1.8345369
SD of NLPD: 0.008867556
Total NLPD: 442.12335
Average loss: 0.0001234211
Total loss: 0.0297444779
Inference type: Laplace approximation full network
Evaluated on: airfoil, Out Of Distribution data
Neural network: [128, 32]
Number of posterior samples: 10000
Average NLPD: 1.827237
SD of NLPD: 0.008953894
Total NLPD: 551.82556
Average loss: 0.0004789172
Total loss: 0.1446329877
```

The Negative Log-Predictive Density (NLPD) is a metric that tells us how 'certain' or 'uncertain' the predictions of the BNN are. For these regression tasks it allows us to compare various NN architectures or Posterior approximation methods.

## Contents

```src:``` Contains source code for creating and training NNs, testing and evaluation routines, metrics.

```data:``` Contains datasets

```notebooks:``` Contains preprocessing of datasets, examples with comments and plots for each type of inference
