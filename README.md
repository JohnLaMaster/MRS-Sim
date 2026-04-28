[![GitHub release (latest by date)](https://img.shields.io/github/v/release/JohnLaMaster/MRS-Sim)](https://github.com/JohnLaMaster/MRS-Sim/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/JohnLaMaster/MRS-Sim)](https://github.com/JohnLaMaster/MRS-Sim/releases)
![GitHub Maintained?](https://img.shields.io/badge/Maintained%3F-yes-brightgreen)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/JohnLaMaster/MRS-Sim?foo=bar)](https://github.com/JohnLaMaster/MRS-Sim/commits/develop)
[![GitHub last commit](https://img.shields.io/github/last-commit/JohnLaMaster/MRS-Sim)](https://github.com/JohnLaMaster/MRS-Sim/commits/develop)
[![License](https://img.shields.io/github/license/JohnLaMaster/MRS-Sim)](https://github.com/JohnLaMaster/MRS-Sim/blob/main/LICENSE.md)
![GitHub python](https://img.shields.io/badge/python-v3-brightgreen.svg)
![GitHub PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-brightgreen.svg)

<div align="center"><img src="https://github.com/JohnLaMaster/MRS-Sim/assets/7785925/7cbee3d1-084f-43e8-80ef-8ae3fee1b0e1" width="400"></div>

# MRS-Sim: Open-Source Framework for Simulating Realistic, In Vivo-like Clinical Magnetic Resonance Spectra

_Abstract:_ 
> Realistic, in vivo-like synthetic data is increasingly needed to develop and validate methods in magnetic resonance spectroscopy. MRS-Sim is a powerful, open-source framework for simulating such data while providing known ground truth values. Its modularity enables modeling the complexities of MRS data for various in vivo scenarios. The underlying physical equations include both commonly used spectral components of linear-combination fitting routines and two novel components. The first is a 3D _B<sub>0</sub>_ field map simulator that models _B<sub>0</sub>_ field inhomogeneities, ranging from slight variations to severe distortions. The second is a novel semi-parametric generator that mimics signals from poorly characterized residual water regions and spectral baseline contributions. This framework can simulate scenarios ranging from raw multi-coil transients to preprocessed, coil-combined multiaverage data. Simulating realistic in vivo-like datasets requires appropriate model parameter ranges and distributions, best determined by analyzing the fitting parameters from existing in vivo data. Therefore, MRS-Sim includes tools for analyzing the ranges and statistical distributions of those parameters from in vivo datasets fitted with Osprey, allowing simulations to be tailored to specific datasets. Additionally, the accompanying repository of supplemental information assists nonexpert users with general simulations of MRS data. The modularity of this framework facilitates easy customization in various in vivo scenarios and promotes continued community development. Using a single framework for diverse applications addresses the inconsistencies in current protocols. By simulating in vivo-like data, MRS-Sim supports many MRS tasks, including verifying spectral fitting protocols and conducting reproducibility analyses. Readily available synthetic data also benefits deep learning research, particularly when sufficient in vivo data is unavailable for training. Overall, MRS-Sim will promote reproducibility and make MRS research more accessible to a wider audience.

# Model Overview
<img src="https://github.com/JohnLaMaster/MRS-Sim/assets/7785925/9b835e36-039a-49d1-aa91-da2adb28071e" width="1000">

# Basis Sets
Currently, two basis sets will be uploaded. They are GE PRESS sequences with a spectral width of 2000, 8192 data points, and TE=30 & 144ms. Additional basis sets will be add occasionally. I am currently switching from metabolite-level basis sets to spin-level basis sets to increase the spectral components that can be modeled. When that is finished, updated basis sets will be provided.

# What makes this model special?
There are several notable differences from standard simulation models. 

    1. Data can be exported in as .mat files AND NIfTI-MRS! The .mat files make it easy to explore the simulated data 
       in Matlab or Jupyter-Notebooks, etc. NIfTI-MRS makes it possible to quantify the simulations easily. More 
       spectral fitting packages are adding NIfTI-MRS compatability.
    2. All the steps in this model are in the opposite order of spectral fitting and processing. Therefore, all the  
       corresponding parameters are the actual correction values you would need to fit the spectra with a standard 
       pipeline. However, when working in conjunction with in vivo data, it is better to fit the synthetic data
       using the same protocol as the in vivo data.
    3. This model aims to always incorporate as many degrees of freedom as possible. The standard variables include 
       amplitudes, Lorentzian and Gaussian lineshapes, (naive) metabolite-level frequency shifts, global frequency 
       shifts, noise, zero- and first-order phase offsets, first-order eddy currents, and baseline signal 
       contributions. Non-standard variables include: residual water contribution, B0 inhomogeneities, and coil 
       artifacts such as phase unalignment, frequency unalignment, sampled coil SNR values, and sampled coil 
       sensitivities. As more improvements to spectral fitting are discovered, they can be added to the model.
    4. On top of a rich baseline modeling, MM/Lip and residual water will be included. Fat/Lip peaks often have a 
       different frequency shift than the water components because they are less sensitive to temperature changes. 
       Therefore, different frequency shifts are applied to the two groups. Separate Gaussian lineshape values are
       also applied to the MM/Lip signal contributions. Originally these signals were included via basis functions 
       from Osprey. However, the updated version of MARSS can now simulate them too and this will be integrated soon.
    5. A variety of quantification methods are built-in. Modulated basis function coefficients, area integration, 
       peak heights, and ratios wrt a metabolite, or combination of metabolites, of choice are produced automatically. 
       These values are usefull for validating new methodologies, but should not be used in conjunction with in vivo
       data as explained above.
    6. Template config files can be used to generate spectra that match those in the publication. An additional DL 
       template is included and is based on my experience. For instance, all (non-creatine) parameters have a 20% 
       chance of being omitted. In my work, this has helped DL models better learn the parameter representations by 
       showing what a sample looks like when the parameter is missing. Additionally, uniform distributions are 
       sampled to present more balanced classes to the networks.

# Examples from the manuscript
The code and output files used to generate the samples for the manuscript are currently being added. Sample simulations are being added to the images folder.  

# How to Contribute?
## Code Base & New Clinical Scenarios
Would you like to add additional functionality to this software? Are you working with types of spectra that are not currently included? Please feel free to make a pull request. Currently, MRS-Sim simulates collections of single voxel spectra. New additions could start with adding paired samples, i.e. a water reference (with eddy current) and a corresponding water suppressed signal or EDIT-ON/EDIT-OFF pairs for J-difference edited spectra. Eventually 2D and other MRS modalities can be developed too. If you already have code that you would like to share, please reach out and we'll see about incorporating it. If you are writing code for this work, please make sure it is VERY well documented so we know how it fits into the repo and what the expected inputs and outputs are.

***Note:** If you happen to find any bugs, please open an issue to let me know. If you have an feature requests, you can open an issue or a pull request.

## Fitting Parameters Dataset Collection
An Open Science Framework project is being put together. The goal of this data repository is to collect the fitting parameters from a variety of clinical scenarios in order to develop a unified prior distribution that will be used to sample model parameters for MRS-Sim. The end goal would be a more automated way to simulate physiologcal or pathology-specific datasets.

## NMR Metabolite Database
Simulation software only works with numerical values. While not the main focus of this work, a database has been compiled with spin definitions and corresponding concentrations and T2 values. I am also looking to include temperature- and pH-induced artifacts and to add X-nuclei as well. If you have literature suggestions or data yourself, please feel free to reach out so we can include it.

# Working with the model
If you use one of the provided configuration templates, please cite the paper below (citation will appear once published). If you would like to collaborate to create a dataset, please reach out at john.t.lamaster (at) gmail dot com.

# Citation
If you use any part of this codebase, please cite the following publication:

> LaMaster, J., Oeltzschner, G., & Li, Y. (2025). MRS‐SIM: Open‐Source Framework for Simulating in Vivo‐Like Magnetic Resonance Spectra. *NMR in Biomedicine, 38*(10), e70130. https://doi.org/10.1002/nbm.70130

<!--
The following is a brief overview:

    1. Fit your data using Osprey and set the flag opts.exportParams.flag=1 and 
       then add the savedir to opts.exportParams.path='path/to/save/dir'
    2. Upload the exported fitting parameters to the repository
Quality control procedures are still being developed. Once the SOP and logistics 
have been finalized, the link to the repo will be posted here. Until then, if 
you are interested in this portion of the project, please feel free to email me.
