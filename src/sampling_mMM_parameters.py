import torch
import numpy as np
import scipy.io as io


__all__ = ['sample_mMM_parameters']


class sample_mMM_parameters(object):
    '''
    This class is meant to sample simulation parameters for simulating mMM signals from Helge Zoellner's
    paper of using cohort-measured MM signals for quantification
    '''
    def __init__(self):
        super().__init__()
        scales = torch.as_tensor([0.9, 1.1])
        # Calculate total Creatine for reference
        # tCr = Cr + PCr
        with open('./src/basis_sets/mMM_parameters_for_simulation.mat','r') as file:
            mat = io.loadmat(file)
        self.cov_matrix = mat['mm_ampl_cov'] * scales
        self.mm_to_tcr_range = mat['mm_to_tcr_range'] * scales
        self.lorentzian_range = mat['lorentzian_range'] * scales
        self.fshift_range = mat['fshift_range'] * scales
        self.double_g_range = mat['double_g_range'] * scales
        
        # Step 2: Partition the mean vector and covariance matrix
        self.mu_MM092 = mat['mu_mm092']
        self.mu_MM = mat['mu_mm']

        self.Sigma_MM092_MM092 = self.cov_matrix[0:1, 0:1]  # Covariance of MM092 (scalar)
        self.Sigma_MM =          self.cov_matrix[1: , 1: ]  # Covariance of MM amplitudes (11x11 matrix)
        self.Sigma_MM092_MM =    self.cov_matrix[0:1, 1: ]  # Covariance between MM092 and MM amplitudes (1x11 matrix)
        self.Sigma_MM_MM092 =    self.cov_matrix[1: , 0:1]  # Covariance between MM amplitudes and MM092 (11x1 matrix)
        
        
    def __call__(self, tCr: torch.Tensor) -> tuple:
        '''
        This function accepts a batch of tCr values and uniformly samples a MM_to_tCr_ratio to calculate an amplitude
        value for MM092. Then conditional means and variances are calculated for the remaining MM basis functions that
        are conditioned on MM092. 
        
        The MM amplitudes are multivariate normal distributed. tCr components are uniformly distributed. the MM to tCr
        ratio, lorentzian, fshifts, and gaussian lineshape are all uniformly sampled.
        
        Inputs:
            :: tCr is the sum of the amplitudes of the creatine containing components
        
        Outputs:
            :: amplitudes of the 11 mMM_updated basis functions
            :: lorentzian lineshape values
            :: secondary MM Gaussian lineshape value
            :: frequency shifts for the 11 basis functions
        '''
        bS = tCr.shape[0]
        # # Define the MM092 amplitude as a function of tCr and a uniform sampling of the ratio range
        # MM092_ampl = tCr * torch.distributions.uniform.Uniform(self.mm_to_tCr_ratio[0], self.mm_to_tCr_ratio[1]).sample(bS)

        # Step 3: Define a batch of independent variable values (tCr) for conditioning
        # Assuming `given_tCr` is a 2D tensor of shape (batch_size, 1) with given tCr values
        # Define the MM092 amplitude as a function of tCr and a uniform sampling of the ratio range
        MM092_ampl = tCr * torch.distributions.uniform.Uniform(self.mm_to_tcr_range[0], self.mm_to_tcr_range[1]).sample(bS)

        # Step 4: Calculate the conditional mean and covariance for each batch of MM amplitudes given tCr
        # Compute the conditional mean and covariance for each batch entry
        mu_MM_given_MM092 = self.mu_MM + (self.Sigma_MM_MM092 * self.Sigma_MM092_MM092.pow(-1)) * (MM092_ampl - self.mu_MM092)
        Sigma_MM_given_MM092 = self.Sigma_MM - (self.Sigma_MM_MM092 * self.Sigma_MM092_MM092.pow(-1)) @ self.Sigma_MM092_MM

        # Step 5: Create a conditional multivariate normal distribution for MM amplitudes for each batch entry
        conditional_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu_MM_given_MM092.squeeze(), Sigma_MM_given_MM092)

        # Step 6: Sample from the conditional distribution for each batch entry
        MM_given_MM092_samples = conditional_dist.sample([1]).squeeze()
        
        d = torch.distributions.uniform.Uniform(low=self.lorentzian_range[0], high=self.lorentzian_range[1]).sample(bS)
        g = torch.distributions.uniform.Uniform(low=self.double_g_range[1,0], high=self.double_g_range[1,1]).sample(bS)
        f = torch.distributions.uniform.Uniform(low=self.fshift_range[0],     high=self.fshift_range[1]).sample(bS)

        return torch.cat([MM092_ampl, MM_given_MM092_samples], dim=-1), d, g, f
    
# I have code from ChatGPT that claims to do the same thing. Here, the data is loaded as the variable mat and MM092_ampl is calculated independently. How does this approach compare with what you just provided?
# self.mu_MM092 = mat['mu_mm092']
# self.mu_MM = mat['mu_mm']

# self.Sigma_MM092_MM092 = self.cov_matrix[0:1, 0:1]  # Covariance of MM092 (scalar)
# self.Sigma_MM =          self.cov_matrix[1: , 1: ]  # Covariance of MM amplitudes (11x11 matrix)
# self.Sigma_MM092_MM =    self.cov_matrix[0:1, 1: ]  # Covariance between MM092 and MM amplitudes (1x11 matrix)
# self.Sigma_MM_MM092 =    self.cov_matrix[1: , 0:1]  # Covariance between MM amplitudes and MM092 (11x1 matrix)

# # Step 4: Calculate the conditional mean and covariance for each batch of MM amplitudes given tCr
# # Compute the conditional mean and covariance for each batch entry
# mu_MM_given_MM092 = self.mu_MM + (self.Sigma_MM_MM092 * self.Sigma_MM092_MM092.pow(-1)) * (MM092_ampl - self.mu_MM092)
# Sigma_MM_given_MM092 = self.Sigma_MM - (self.Sigma_MM_MM092 * self.Sigma_MM092_MM092.pow(-1)) @ self.Sigma_MM092_MM

# # Step 5: Create a conditional multivariate normal distribution for MM amplitudes for each batch entry
# conditional_dist = torch.distributions.multivariate_normal.MultivariateNormal(mu_MM_given_MM092.squeeze(), Sigma_MM_given_MM092)

# # Step 6: Sample from the conditional distribution for each batch entry
# MM_given_MM092_samples = conditional_dist.sample([1]).squeeze()