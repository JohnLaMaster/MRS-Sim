# In-Vivo-MRSI-Simulator
Notebook and Physics model to generate in vivo MRSI data to be used for deep learning

# What makes this model special?
There are several notable differences from standard models. 
    1. All the steps in this model are in the opposite order of spectral fitting and processing. Therefore, all the corresponding parameters are the actual correction values you would need to fit the spectra with a standard pipeline.
    2. The generator is specifically geared towards facilitating deep learning projects. While all of the model parameters are realistic values (physiological or pathological), the various combinations may be non-sensical from a spectroscopy point of view. However, DL models train better with uniform representations in the training data. Class balanace is thereby ensured by using uniform distributions when sampling the parameters.
    3. All (non-creatine) parameters have a 20% chance of being omitted. This helps DL models better learn the parameter representations by showing what a sample looks like when the parameter is missing.
    4. This model now incorporates a few more degrees of freedom than most fitting models. For example, in clinical data the space between NAA and Cr sometimes has very slight variations. This has been included in the model. Secondly, non-perfectly uniform magnetic fields are now accounted for as well by modeling B0 inhomogeneities. As more improvements to spectral fitting are discovered, they will be added to the model.
    5. On top of a rich baseline modeling, MM/Lip and (hopefully!) residual water will be included. Fat/Lip peaks often have a different frequency shift than the water components because they are less sensitive to temperature changes. Therefore, different frequency shifts are applied to the two groups. 
    6. A variety of quantification methods are built-in. Modulated basisline area integration, basisline peak heights, and ratios wrt Cr are produced automatically.
