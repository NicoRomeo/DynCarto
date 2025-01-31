# DynCarto
Contrastive Cartography for dynamical data analysis

This code supports the paper "Characterizing nonlinear dynamics using contrastive learning" 
by Nicolas Romeo, Chris Chi, Aaron R. Dinner, Elizabeth R. Jerison

Pre-trained cartographer neural networks are available in `/cartographers`.

## Environments

Differential equation integration and analysis are done in different computing environments,
with hdf5 files used to bridge the two.

Dynamical system integration is done in Julia, using `DifferentialEquations.jl`, with the exception of the
the 2-dimensional channel flow simulations. These are in python and require `FEniCSx` [link ]. 
See their website for installation instruction 

The learning and analysis code is in python and reliant on `pytorch`.

Learning code uses `pytorch`. Data handling uses `h5py`  `tqdm` used throughout. (get environment file)
The learning code in particular is GPU-compatible and as written works in a CUDA-enabled environment.




