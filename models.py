import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from prob_dists import *

from pytorch_model_summary import summary

# importing distributions
import torch.distributions as dists

# initialized within VAE class 
class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__() # init parent (nn.module)
        # encoder_net: torch.Sequential
        self.encoder = encoder_net

    # VAE reparameterization trick 
    @staticmethod
    def reparameterization(mu, log_var):
        """
        Instead of sampling z directly, which would make backprobagation impossible, 
        epsilon is sampled instead and z is calculated. 
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)

        return mu + std * eps # = z

    def encode(self, x):
        # output of encoder network
        h_e = self.encoder(x)

        # splitting into 2 equal sized chunks
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e

    # Sampling z through reparameterization trick
    def sample(self, x=None, mu_e=None, log_var_e=None):
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-var can`t be None!')
        z = self.reparameterization(mu_e, log_var_e)
        return z

    # sampling log probability
    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-var and z can`t be None!')

        return log_normal_diag(z, mu_e, log_var_e)

    # forward returns log probability
    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='categorical', num_vals=None):
        super(Decoder, self).__init__()

        self.decoder = decoder_net
        self.distribution = distribution
        self.num_vals = num_vals

    def decode(self, z):

        # input are latent variables, 2*L (mean and variance)
        # the output depends on expected output distribution, see below.
        h_d = self.decoder(z) # node: 'decode' and 'decoder' to minimize confusion
        # hidden output of decoder

        if self.distribution == 'categorical':
            # Categorical distribution has multiple possible outputs
            # output dim: (batch, D*L values) where D are all outputs and L is the number of possible values per output
            #   - e.g. D are image dimension and L are possible pixel values
            b = h_d.shape[0] # B, batch size
            d = h_d.shape[1] // self.num_vals # D, (D*L) // L = D, how often L goes up to D*L
            h_d = h_d.view(b, d, self.num_vals) # reshaping to (B, D, L)
            # softmax over last dimension L
            #   - e.g. probs for pixel values for each pixel in img
            prob_d = torch.softmax(h_d, 2)
            # probability output of decoder
            return [prob_d]

        elif self.distribution == 'bernoulli':
            # Bernoulli distribution has two possible outcomes
            # output dim: (batch, D*1), where the are outputs and 1 is the single output probability
            prob_d = torch.sigmoid(h_d)
            return [prob_d]

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

    def sample(self, z):
        prob_d = self.decode(z)[0] # probability output

        if self.distribution == 'categorical':
            b = prob_d.shape[0] # batch size
            m = prob_d.shape[1] # output dimension
            # below is unnecessary because already performed in self.decode()
            #prob_d = prob_d.view(prob_d.shape[0], -1, self.num_vals) # -1 is inferred from other dims (what is left)
            p = prob_d.view(-1, self.num_vals) # merging batch and output dims (lists of possible outputs)
            # we want one sample per number of possible values (e.g. pixel values)
            x_new = torch.multinomial(p, num_samples=1).view(b, m) # new view is (batch, output dims, 1)

        elif self.distribution == 'bernoulli':
            x_new = torch.bernoulli(prob_d) # output dim is already (batch, output dims, 1)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return x_new


    def log_prob(self, x, z):
        # calculating the log−probability which is later used for ELBO
        prob_d = self.decode(z)[0] # probability output
        if self.distribution == 'categorical':
            log_p = log_categorical(x, prob_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.distribution == 'bernoulli':
            log_p = log_bernoulli(x, prob_d, reduction='sum', dim=-1)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return log_p

    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(x)


class Prior(nn.Module):
    def __init__(self, L):
        super(Prior, self).__init__()
        self.L = L

    def sample(self, batch_size):
        z = torch.randn((batch_size, self.L))
        return z

    def log_prob(self, z):
        return log_standard_normal(z)


class VAE(nn.Module):
    def __init__(self, encoder_net, decoder_net, num_vals=256, L=16, likelihood_type='categorical'):
        super(VAE, self).__init__()


        self.encoder = Encoder(encoder_net=encoder_net)


        self.decoder = Decoder(distribution=likelihood_type, decoder_net=decoder_net, num_vals=num_vals)
        self.prior = Prior(L=L)

        self.num_vals = num_vals

        self.likelihood_type = likelihood_type

    def forward(self, x, reduction='avg'):
        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # ELBO
        # reconstruction error
        RE = self.decoder.log_prob(x, z) # z is decoded back
        # Kullback–Leibler divergence, regularizer
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)
        # loss
        if reduction == 'sum':
            return -(RE + KL).sum()
        else:
            return -(RE + KL).mean()

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)




