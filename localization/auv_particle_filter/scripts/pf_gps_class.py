#!/usr/bin/env python3

import numpy as np
from auvlib.data_tools import gsf_data, std_data, csv_data, xyz_data
from auvlib.bathy_maps import mesh_map, base_draper
import configargparse
import math
import os
import sys
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from scipy.stats import norm
from scipy.spatial.transform import Rotation as rot
import torch
import gpytorch
import time

# import tensorflow as tf # For the KL Divergence

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        # self.covar_module = gpytorch.kernels.LinearKernel() # Doesn't work...
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP_mbes:
    # -------------- Global constants ------------------------------------:
    # MBES parameters
    beam_width = math.pi/180.*60.
    nbr_beams = 512
    # ---------------------------------------------------------------------

    def __init__(self, mbes_meas_ranges, sim_meas_ranges):
        self.mbes_meas_ranges = mbes_meas_ranges  
        self.sim_meas_ranges = sim_meas_ranges 

    def Training_data(self):
        # Beams distributed along seafloor - AUV's y axis
        self.train_x_mbes = torch.Tensor([*range(len(self.mbes_meas_ranges))])
        self.train_x_sim = torch.Tensor([*range(len(self.sim_meas_ranges))])
        # Depth data - AUV's z axis
        self.train_y_mbes = torch.Tensor(self.mbes_meas_ranges)
        self.train_y_sim = torch.Tensor(self.sim_meas_ranges)

    # Run regression (all lumped into one function for now)
    def run_gpytorch_regression(self, train_x, train_y): 
        """
        Run regression (all lumped into one function for now)
        """
        # %matplotlib inline
        # %load_ext autoreload
        # %autoreload 2

        # -------------- Constants ------------------------------------:
        training_point_count = 15
        max_training_iter_count = 300
        noise_threshold = 0.01 # Use noise to determine when to stop training
        loss_threshold = -2.0
        # -------------------------------------------------------------
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        # this is for running the notebook in our testing framework
        import os
        smoke_test = ('CI' in os.environ)
        max_train_iter = 2 if smoke_test else max_training_iter_count

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        i = 0 # init iteration count
        loss_val = 999. # init loss value for convergence w/ large value

        # Train until convergence
        # while loss_val >= loss_threshold:
        while model.likelihood.noise.item() >= noise_threshold:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            if (i+1) % 10 == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.4f' % (
                    i + 1, max_train_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))
            optimizer.step()

            # Update iteration / convergence tracking
            loss_val = loss.item()
            i += 1
            if i > max_train_iter:
                break

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # Test points are regularly spaced on range of y-axis
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():  # To use LOVE 
            test_x = torch.linspace(min(train_x), max(train_x), training_point_count)
            observed_pred = likelihood(model(test_x))

        return test_x, observed_pred


    def KLgp_div(self):
        # Calculate KL divergence
        cov_mbes = self.observed_pred_mbes.lazy_covariance_matrix.numpy()
        # cov_mbes = self.observed_pred_mbes.covariance_matrix.detach().numpy()
        mu_mbes  = self.observed_pred_mbes.mean.numpy()
        cov_sim  = self.observed_pred_sim.lazy_covariance_matrix.numpy()
        # cov_sim  = self.observed_pred_sim.covariance_matrix.detach().numpy()
        mu_sim   = self.observed_pred_sim.mean.numpy()

        # print('\n cov_sim: \n {}'.format(np.array_str(cov_sim,  precision=3)))
        # print('\n cov_mbes:\n {}'.format(np.array_str(cov_mbes, precision=3)))

        # print('\ndet(cov_sim):  {}'.format(np.linalg.det(cov_sim)))
        # print(  'det(cov_mbes): {}'.format(np.linalg.det(cov_mbes)))

        def is_pos_semi_def(x):
            return np.all(np.linalg.eigvals(x) >= 0)

        # print("\nPositive semi-def check: cov_sim: ",  is_pos_semi_def(cov_sim))
        # print(  "Positive semi-def check: cov_mbes: ", is_pos_semi_def(cov_mbes))

        dim = cov_mbes.shape[0] # Dimension
        mu_sub = np.array([mu_sim - mu_mbes])
        kl1 = np.trace(np.dot(np.linalg.inv(cov_sim),cov_mbes))
        kl2 = np.dot(np.dot(mu_sub,np.linalg.inv(cov_sim)), np.transpose(mu_sub))[0,0]
        kl3 = np.log(np.linalg.det(cov_sim) / np.linalg.det(cov_mbes))

        # print("\nkl1: ", kl1)
        # print(  "kl2: ", kl2)
        # print(  "kl3: ", kl3)
        # print(  "dim: ", dim)

        # Catch for when det(cov_sim) and/or det(cov_mbes) = 0.0
        if math.isnan(kl3):
            kl_div = 0.5*(kl1 + kl2 - dim)
        else:
            kl_div = 0.5*(kl1 + kl2 - dim + kl3)

        # print('\n KL divergence: {}'.format(kl_div, precision=3))
        return 1 / kl_div

