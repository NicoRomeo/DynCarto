"""Contrastive learning neural network architecture

Defines a Multilayer perceptron  and the I_estimator class which implements contrastive learning using
a single-encoder InfoNCE architecture.

"""

import numpy as np
import torch
import torch.nn as nn
#from nn_utils import MLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

class MLP(nn.Module):
    """A class to define a neural network in pytorch"""
    def __init__(self, channels, activated=False, activation=nn.ReLU(), dropout=0):
        super().__init__()

        n_layer = len(channels) - 1
        layers = []

        for i in range(n_layer):
            in_feats, out_feats = channels[i], channels[i+1]
            layers.append( nn.Linear(in_feats, out_feats) )
            if i < n_layer-1:
                layers.append( activation )
                if dropout: layers.append( nn.Dropout(dropout) )
                
        if activated and n_layer > 0: 
            layers.append( activation ) 

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class I_estimator(nn.Module):
    """InfoNCE based contrastive learning pipeline"""
    def __init__(self,
                input_shapes, # (int, int)
                n_layers, # int
                n_channels, # int
                LR, # Learning rate, scheduling, etc.
                ):
        
        super().__init__()

        self.make_encoder(input_shapes, n_layers, n_channels, activation='relu') #n_ch, n_layers, strides_enc, dilations_enc
        self.optimizer = torch.optim.Adam([{'params': self.parameters()}],
                                             lr=LR)

    def make_encoder(self, input_shapes, n_layers, n_channels, activation='tanh'):
        channels = [n_channels,]*n_layers
        self.x_embedder = MLP([input_shapes[0], *channels], activation) 
        self.y_embedder = MLP([input_shapes[1], *channels], activation) 
        return

    def forward(self, x, y):
        enc1 = self.x_embedder(x) # x is size batch x dimension of feature ; enc1 is  batch x embedding dim
        enc2 = self.x_embedder(y)

        f = torch.einsum('ij,kj->ik', enc1, enc2) # measures cosine distances, size batch x batch
        #print(f.shape)
        
        # The loss function computes a cross entropy over positive/negative samples. 
        # positive samples are on diagonal, negative samples are all others.
        loss = torch.nn.functional.cross_entropy(f, 
                                                 torch.eye(len(f), device=device),
                                                 reduction='mean') - np.log(f.shape[0])
        # the eye is the true matrix, encoding which pairs of samples are positive. f ~ log(p)
        # - np.log(f.shape[0]) = - log of batch size : makes loss an estimate to -(mutual information)
        return loss, enc1, enc2, f
