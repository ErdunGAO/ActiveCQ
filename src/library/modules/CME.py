"""[summary]
This is some code running for BayesCME to test for hyperparameter learning

- The idea is to express CME as a vector valued GP regression and learn the hyperparameter accordingly
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from gpytorch.kernels import RBFKernel, MaternKernel, RQKernel
from sklearn.metrics import pairwise_distances

from torch.optim import Adam

####################
# Helper Functions #
####################

def median_heuristic(x):
    """[One dimensional Median Heuristic]

    Args:
        x ([torch tensor]): [One dimensional torch tensor]

    Returns:
        [type]: [lengthscale obtained using median heuristic]
    """
    x_np = x.numpy()
    dists = pairwise_distances(x_np)
    median_dist = np.median(dists)
    if median_dist == 0.0:
        # Using a lengthscale of 0 can lead to NaNs, especially with gpytorch's constraints.
        # This is common for discrete/binary inputs. We default to 1.0 for stability.
        return 1.0
    return median_dist

#############################
# Learning of CMEs #
#############################
class CME_learner(nn.Module):

    def __init__(self, 
                x,
                y,
                kernel_x="rbf",
                kernel_y="rbf",
                lambda_init=torch.Tensor([1]).double(),
                device="cpu"):
        """[Bayesian CME]

        Args:
            x ([tensor]): [conditioned variable]
            y ([tensor]): [target variable]
            lambda_init ([tensor], optional): [description]. Defaults to torch.Tensor([1]).float().
        """
        
        super().__init__()
        # Setup
        # self.x, self.y = x, y
        self.device = device
        self.x = x.to(self.device).double()
        self.y = y.to(self.device).double()

        # Initialise Kernels
        #NOTE: here we usually only condition on one variable
        if kernel_x == "rbf":
            self.k_x = RBFKernel(ard_num_dims=x.shape[1])
        elif kernel_x == "matern":
            self.k_x = MaternKernel(ard_num_dims=x.shape[1])
        elif kernel_x == "rq":
            self.k_x = RQKernel(ard_num_dims=x.shape[1])
        else:
            raise ValueError("Kernel not supported")
        

        #NOTE: here we usually fix the parameters.
        if kernel_y == "rbf":
            self.l_y = RBFKernel(ard_num_dims=y.shape[1])
        elif kernel_y == "matern":
            self.l_y = MaternKernel(ard_num_dims=y.shape[1])
        elif kernel_y == "rq":
            self.l_y = RQKernel(ard_num_dims=y.shape[1])
        else:
            raise ValueError("Kernel not supported")

        self.lmda = torch.nn.Parameter(lambda_init)

        # Initialise Kernel with median heuristic hyperparameters
        self.k_x.lengthscale = torch.Tensor([median_heuristic(x)]).double()
        self.l_y.lengthscale = torch.Tensor([median_heuristic(y)]).double()

    def forward(self):

        # Compute the kernel, outputting lazy kernels
        K = self.k_x(self.x).double()
        L = self.l_y(self.y).double()

        # Compute K + lambda^2 I (square to ensure positiveness)
        K_lambda = K.add_diag(self.lmda**2)

        # Return the componenets required to evaluate the Likelihood
        return K_lambda, L


# Need to define a likelihood function as loss

def nll(K_lambda, L):
    """[Computes the negative loglikelihood of the CME as GP regression ]

    Args:
        K_lambda ([lazy kernel]): [Kernel matrix of the conditioned variable + regularisation]
        L ([lazy kernel]): [Kernel Matrix of the target variable]

    Returns:
        [torch tensor float]: [the negative log likelihood]
    """

    return K_lambda.logdet() + torch.trace(K_lambda.inv_matmul(L.evaluate()))
