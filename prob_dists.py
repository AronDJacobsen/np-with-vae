import torch
import numpy as np
import torch.nn.functional as F
import math
from torch import nn, Tensor
from torch.distributions import Distribution

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-5

def log_categorical(x, p, reduction='sum', dim=None):
    #x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    #log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    # using clamp to avoid log(0) by setting min and max values as EPS and 1-EPS
    log_p = x * torch.log(torch.clamp(p, EPS, 1. - EPS))

    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_categorical_natural(x, p, reduction='sum'):
    log_p = x * torch.log(torch.clamp(p, EPS, 1. - EPS))

    return log_p


def categorical(x, p, num_classes=256, reduction=None, dim=None):
    #x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    #p = x_one_hot * (torch.clamp(p, EPS, 1. - EPS))
    p = x * (torch.clamp(p, EPS, 1. - EPS))

    if reduction == 'avg':
        return torch.mean(p, dim)
    elif reduction == 'sum':
        return torch.sum(p, dim)
    else:
        return p

def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_normal(x, mu, log_var, reduction=None, dim=None):
    #log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    # the same:
    log_p = torch.distributions.normal.Normal(mu, torch.sqrt(torch.exp(0.5*log_var))).log_prob(x)

    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_normal_natural(x, eta1, eta2, reduction=None, dim=None):
    #log_p = torch.distributions.normal.Normal(mu, torch.sqrt(torch.exp(0.5*log_var))).log_prob(x)
    #torch.cholesky(eta1)
    log_p = torch.log(torch.tensor([1]) / torch.sqrt(2*PI)) + eta1*x + eta2*(x**2) + (eta1**2)/4*eta2 + torch.log(-2*eta2)/2
    '''
    precmu   = g.theta1
    sqrtprec = chol(-g.theta2)
    tmp      = sqrtprec'\precmu
    fun(x)   = ( dot(x,g.theta2*x) + 2dot(x,precmu) )/2
    result   = sum(neghalflog2pi + log.(diag(sqrtprec))) - norm(tmp)^2/2
    result  += funarray(fun, x, length(precmu), axis)
    '''
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


# TODO: log normal per class as well (how likely is mu wrt. a mu/sigma)
def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    # expect mean=0 and sigma=1
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

### from Deep Learning:
class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, log_sigma: Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        # raise NotImplementedError # <- your code
        return torch.distributions.normal.Normal(self.mu, self.sigma).rsample()

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        # raise NotImplementedError # <- your code
        return torch.distributions.normal.Normal(self.mu, self.sigma).log_prob(z)