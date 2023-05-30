# <img src="https://github.com/JohnLaMaster/MRS-Sim/assets/7785925/c9563f8a-0034-4edd-bf20-551dbe86c1ac" width="100"> MRS-Sim: Open-Source Framework for Simulating Realistic, In Vivo-like Clinical Spectra
**In review: Expected publication in _Summer 2023_.**

***Note:** Code is still in development and is expected to be finalized by the date of publication. Be sure to check back during the summer 2023 for the first official release.

_Abstract:_ This work presents the first open-source, publicly available \textit{in-vivo}-like data simulator for generating synthetic magnetic resonance spectroscopy data that is geared towards community development. This simulator uses high fidelity basis functions that are simulated for sequence- and brand-specific scenarios. The underlying physics model includes all spectral components found in spectral fitting routines, i.e. zero- and first-order phase, Voigtian lineshapes, frequency shifts, and eddy currents. In addition to the standard spectra types, this model can simulate multi-coil transients with or without coil artifacts as well as raw and pre-processed spectra. A $B_0$ field map simulator can be used to model severe susceptibility effects commonly found near the sinuses or deep brain structures. Finally, a novel semi-parametric simulator for residual water regions and baseline offsets provides in vivo-like artifacts. Accompanying software can analyze the distributions and ranges of parameters in fitted datasets which allows researchers to tailor synthetic datasets to their clinical ones.

# Model Overview
<img src="https://github.com/JohnLaMaster/MRS-Sim/assets/7785925/9b835e36-039a-49d1-aa91-da2adb28071e" width="1000">


# Basis Sets
Currently, two basis sets will be uploaded. They are GE PRESS sequences with a spectral width of 2000, 8192 data points, and TE=30 & 144ms. Additional basis sets will be simulated and uploaded for Siemens and Philipps.

# What makes this model special?
There are several notable differences from standard simulation models. 

    1. All the steps in this model are in the opposite order of spectral fitting and processing. Therefore, all the  
       corresponding parameters are the actual correction values you would need to fit the spectra with a standard pipeline.
    2. This model now incorporates a few more degrees of freedom than most fitting models. Standard variables include 
       amplitudes, Lorentzian and Gaussian lineshapes, (naive) metabolite-level frequency shifts, global frequency shifts, noise,
       zero- and first-order phase offsets, first-order eddy currents, and baseline signal contributions. Non-standard variables 
       include: residual water contribution, B0 inhomogeneities, and coil artifacts such as phase unalignment, frequency 
       unalignment, sampled coil SNR values, and sampled coil sensitivities. As more improvements to spectral fitting are 
       discovered, they can be added to the model.
    3. On top of a rich baseline modeling, MM/Lip and residual water will be included. Fat/Lip peaks often have  a
       different frequency shift than the water components because they are less sensitive to temperature changes. Therefore, 
       different frequency shifts are applied to the two groups. MM/Lip signal contributions are included via basis functions
       from Osprey.
    4. A variety of quantification methods are built-in. Modulated basis function coefficients, area integration, peak heights, and  
       ratios wrt Cr are produced automatically.      
    5. Template config files can be used to generate spectra that match those in the publication. An additional DL template is
       included and is based on my experience. For instance, all (non-creatine) parameters have a 20% chance of being omitted. In
       my work, this has helped DL models better learn the parameter representations by showing what a sample looks like when the 
       parameter is missing. Additionally, uniform distributions are sampled to present more balanced classes to the networks.

# Examples from the manuscript
The code and output files used to generate the samples for the manuscript are currently being added. Sample simulations are being added to the images folder.  

# How to Contribute?
## Code Base
Would you like to add additional functionality to this software? Are you working with types of spectra that are not currently included? Please feel free to make a pull request. If you already have code that you would like to share, please reach out and we'll see about incorporating it. If you are writing code for this work, please make sure it is VERY well documented so we know how it fits into the repo and what the expected inputs and outputs are.

## Fitting Parameters Dataset Collection
An Open Science Framework project is currently being put together. The goal of this data repository is to collect the fitting parameters from a variety of clinical scenarios in order to develop a unified prior distribution that will be used to sample model parameters for MRS-Sim. The following is a brief overview:
    1. Fit your data using Osprey and set the flag opts.exportParams.flag=1 and 
       then add the savedir to opts.exportParams.path='path/to/save/dir'
    2. Upload the exported fitting parameters to the repository
Quality control procedures are still being developed. Once the SOP and logistics 
have been finalized, the link to the repo will be posted here. Until then, if 
you are interested in this portion of the project, please feel free to email me.

# Working with the model
If you use one of the provided configuration templates, please cite the following paper (citation will appear once published). If you would like to collaborate to create a dataset, please reach out at john.t.lamaster (at) gmail dot com.
