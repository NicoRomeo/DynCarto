"""
Some analysis tools to study the learned representation of dynamical systems

"""

import numpy as np
from sklearn import decomposition, preprocessing
from sklearn import manifold
from sklearn.decomposition import PCA
import h5py
import torch


def load_model(filename):
    """ Loads the cartographer onto the cpu for use 
    Parameters:
    filename (str): location of the model .pt file
    """
    return torch.load(filename, map_location=torch.device('cpu'))
    
def get_dist_matrix_fhn(filename, model, num_reads, stride, fromjl=True):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        num_perclass = num_systems // 4
        list_x1s = []
        parameters = []
        for j in range(4):
            for i in range(num_reads):
                if fromjl:
                    idx = str(j*num_perclass + i+1)
                    xi = torch.permute(
                        torch.tensor(np.array(f[idx][:,::stride,:], dtype=np.float32)),
                        (2,0,1)
                        )
                    xi = (xi - torch.mean(xi))/torch.std(xi)
                    parameters.append(j)
                else:
                    idx = str(j*num_perclass + i)
                    xi = torch.tensor(np.array(f[idx][:,::stride,:], dtype=np.float32))
                    
                    xi = (xi - torch.mean(xi))/torch.std(xi)
                    parameters.append(f[idx].attrs["params"])
                list_x1s.append(xi)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)
        
        _, _, _, f  = model(x1, x1)
        return f.detach().numpy(), parameters
    
def get_dist_matrix(filename, model, num_reads=0,strides=1, num_IC=-1, fromjl=False):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        if num_reads == 0:
            num_reads = num_systems
        else:
            num_reads = min(num_systems, num_reads)
        list_x1s = []
        parameters = []
        for i in range(num_reads):
            if fromjl:
                idx = str(i+1)
                xi = torch.permute(
                    torch.tensor(np.array(f[idx], dtype=np.float32)[:,::strides,:]), 
                    (2,1,0))
                parameters.append(np.swapaxes(f[idx].attrs["params"],1,0))
            else:
                idx = str(i)
                xi = torch.tensor(np.array(f[idx], dtype=np.float32)[:,::strides,:])
                if num_IC > 0:
                    if xi.shape[0] != num_IC:
                        continue
                parameters.append(f[idx].attrs["params"])
            xi = (xi - torch.mean(xi))/torch.std(xi)
            list_x1s.append(xi)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)
        _, _, _, f  = model(x1, x1)
        return f.detach().numpy(), parameters


def PCA_on_dists(dists):
    dists_scaled = preprocessing.scale(dists, axis=1)
    pca = decomposition.PCA(n_components=7)
    pca.fit(dists_scaled)
    X_pca = pca.transform(dists_scaled)
    return X_pca, pca.explained_variance_ratio_, pca
    
def get_latent_fhn(filename, model, num_reads, stride, fromjl=True):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        num_perclass = num_systems // 4
        list_x1s = []
        parameters = []
        for j in range(4):
            for i in range(num_reads):
                if fromjl:
                    idx = str(j*num_perclass + i+1)
                    xi = torch.permute(
                        torch.tensor(np.array(f[idx][:,::stride,:], dtype=np.float32)),
                        (2,0,1)
                        )
                    xi = (xi - torch.mean(xi))/torch.std(xi)
                    parameters.append(j)
                else:
                    idx = str(j*num_perclass + i)
                    xi = torch.tensor(np.array(f[idx][:,::stride,:], dtype=np.float32))
                    parameters.append(f[idx].attrs["params"])
                list_x1s.append(xi)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)
        
        _, enc1, _, f  = model(x1, x1)
        return enc1.detach().numpy(), parameters

def get_latent(filename, model, num_reads=0, num_start=0, strides=1, num_IC=-1, fromjl=False):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        if num_reads == 0:
            num_reads = num_systems-num_start
        else:
            num_reads = min(num_systems-num_start, num_reads)
        dists = np.zeros((num_reads, num_reads))
        list_x1s = []
        list_x2s = []
        parameters = []
        for i in range(num_start, num_reads+num_start):
            if fromjl:
                idx = str(i+1)
                xi = torch.permute(
                    torch.tensor(np.array(f[idx], dtype=np.float32)[:,::strides,:]), 
                    (2,1,0))
                parameters.append(np.swapaxes(f[idx].attrs["params"],1,0))
            else:
                idx = str(i)
                xi = torch.tensor(np.array(f[idx], dtype=np.float32)[:,::strides,:])
                if num_IC > 0:
                    if xi.shape[0] != num_IC:
                        continue
                parameters.append(f[idx].attrs["params"])
            xi = (xi - torch.mean(xi))/torch.std(xi)
            list_x1s.append(xi)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)
        
        _, enc1, _, f  = model(x1, x1)
        return enc1.detach().numpy(), parameters
    
def get_latent2(filename, model, num_reads=0, strides=1, num_IC=-1):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        if num_reads == 0:
            num_reads = num_systems
        else:
            num_reads = min(num_systems, num_reads)
        list_x1s = []
        parameters = []
        for i in range(1,num_reads+1):
            idx = str(i)
            xi = torch.tensor(np.array(f[idx], dtype=np.float32)[:,::strides,:])
            if num_IC > 0:
                if xi.shape[0] != num_IC:
                    continue
            parameters.append(f[idx].attrs["params"])
            xi = (xi - torch.mean(xi))/torch.std(xi)
            list_x1s.append(xi)
        x1 = torch.flatten(torch.stack(list_x1s, dim=0),start_dim=1)
        
        _, enc1, _, f  = model(x1, x1)
        return enc1.detach().numpy(), parameters
    
def PCA_on_latent(latents):
    latents_scaled = preprocessing.scale(latents, axis=1)
    pca = decomposition.PCA(n_components=7)
    pca.fit(latents_scaled)
    X_pca = pca.transform(latents_scaled)
    return X_pca, pca.explained_variance_ratio_, pca


def get_trajs(filename, num_reads=0, strides=1, num_IC=-1, fromjl=False):
    with h5py.File(filename, 'r') as f:
        num_systems = len(f.keys())
        if num_reads == 0:
            num_reads = num_systems
        else:
            num_reads = min(num_systems, num_reads)
        #dists = np.zeros((num_reads, num_reads))
        list_x1s = []
        #list_x2s = []
        #parameters = []
        for i in range(num_reads):
            if fromjl:
                idx = str(i+1)
                xi = torch.permute(
                    torch.tensor(np.array(f[idx], dtype=np.float32)[:,::strides,:]), 
                    (2,1,0))
                #parameters.append(np.swapaxes(f[idx].attrs["params"],1,0))
            else:
                idx = str(i)
                xi = torch.tensor(np.array(f[idx], dtype=np.float32)[:,::strides,:])
                if num_IC > 0:
                    if xi.shape[0] != num_IC:
                        continue
                #parameters.append(f[idx].attrs["params"])
            xi = (xi - torch.mean(xi))/torch.std(xi)
            list_x1s.append(xi)
        return list_x1s
    