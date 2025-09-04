# DynCarto
Contrastive Cartography for dynamical data analysis

This code supports the paper "Characterizing nonlinear dynamics using contrastive learning" 
by Nicolas Romeo, Chris Chi, Aaron R. Dinner, Elizabeth R. Jerison

Pre-trained cartographer neural networks are available in `/cartographers`.

## Environments

Differential equation integration and analysis are done in different computing environments,
with hdf5 files used to bridge the two.

Dynamical system integration is done in Julia, using [`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable/), with the exception of the
the 2-dimensional channel flow simulations. These are in python and require [`FEniCSx`](https://fenicsproject.org). 
See their website for installation instruction 

The learning and analysis code is in python and reliant on [`pytorch`](https://pytorch.org).

Learning code uses [`pytorch`](https://pytorch.org). Data handling uses [`h5py`](https://www.h5py.org), and  [`tqdm`](https://github.com/tqdm/tqdm) is used throughout.
The learning code in particular is GPU-compatible and as written works in a CUDA-enabled environment.




