import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from prob_dists import *

# importing distributions
import torch.distributions as dists


def to_natural(prob_d):
    mu = prob_d[:, 0]
    sigma = prob_d[:, 1]

    eta2 = -0.5 / sigma ** 2
    eta1 = -2 * mu * eta2

    return torch.stack((eta1, eta2), dim=1)


# initialized within VAE class
class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()  # init parent (nn.module)
        # encoder_net: torch.Sequential
        self.encoder = encoder_net

    # VAE reparameterization trick
    @staticmethod
    def reparameterization(mu, log_var):
        """
        Instead of sampling z directly, which would make backprobagation impossible,
        epsilon is sampled instead and z is calculated.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps  # = z

    def encode(self, x):
        # output of encoder network
        h_e = self.encoder(x.float()) # Output: one Mu and Sigma per numerical and categorical variable

        # splitting into 2 equal sized chunks - wherein one will be trained towards mu the other towards log-var
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

        return log_normal(z, mu_e, log_var_e)

    # forward returns log probability
    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)

class Decoder(nn.Module):
    def __init__(self, decoder_net, var_info, total_num_vals=None, natural=True, scale='none', device=None):
        super(Decoder, self).__init__()

        self.decoder = decoder_net
        self.var_info = var_info
        self.total_num_vals = total_num_vals  # depends on num_classes of each attribute
        self.natural = natural
        self.scale = scale
        self.device = device
        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus()
        # self.distribution = 'gaussian'

    def decode(self, z):

        # input are latent variables, 2*L (mean and variance)
        # the output depends on expected output distribution, see below.
        h_d = self.decoder(z)  # node: 'decode' and 'decoder' to minimize confusion

        if self.natural:
            params = h_d.clone()
        else:
            params = torch.zeros(h_d.shape)

        # hidden output of decoder
        idx = 0
        # decoder outputs the distribution parameters (e.g. mu, sigma, eta's)
        # if not self.natural:

        # Putting constrictions on parameters! 
            # Natural gets softplus on Eta2
            # Regular gets softmax'ed on Sigma
        for var in self.var_info:
            if self.var_info[var]['dtype'] == 'categorical':
                num_vals = self.var_info[var]['num_vals']
                # if not naturals
                if not self.natural:
                    params[:, idx:idx + num_vals] = self.softmax(h_d[:, idx:idx + num_vals])

                idx += num_vals
            elif self.var_info[var]['dtype'] == 'numerical':
                # gaussian always outputs two values
                num_vals = 2
                # if not naturals
                if not self.natural:
                    # normal distribution, mu and sigma returned
                    params[:, idx:idx + num_vals] = h_d[:, idx:idx + num_vals]
                    # todo???

                else:
                    # eta2 have to be negative -inf<eta2<0
                    # extracting eta2
                    params[:, idx+1:idx+2] = -self.softplus(h_d[:, idx+1:idx+2])

                idx += num_vals
            else:
                raise ValueError('Either `categorical` or `gaussian`')

        # returning probability distribution or returning the etas
        return params

    def sample(self, params):
        # TODO: cannot do flatten if z is batched
        #output = self.decode(z)  # probability output
        # prob_d has [mu1, sigma1, mu2, sigma2, ...]
        total_numvals = sum(
            [self.var_info[var]['num_vals'] if self.var_info[var]['dtype'] == 'categorical' else 1 for var in
             self.var_info.keys()])
        x_reconstructed = torch.zeros(params.shape[0], total_numvals)
        # for batch in range(output.size()[0]):
        # var_index = 0
        output_idx = 0
        recon_idx = 0
        for x_idx, var in enumerate(self.var_info):
            num_vals = self.var_info[var]['num_vals']

            if self.var_info[var]['dtype'] == 'categorical':

                if self.natural:  # note that outputs are just logits of probability
                    outs = self.softmax(params[:, output_idx:output_idx + num_vals])
                else:
                    outs = params[:, output_idx:output_idx + num_vals]

                # TODO: output and x_recon can't be accessed similarly.
                outs = outs.view(outs.shape[0], -1, num_vals)
                p = outs.view(-1, num_vals)

                x_batch = torch.multinomial(p, num_samples=1)  # .view(b, m) # new view is (batch, output dims, 1)

                # one hot encoding x_batch:
                x_batch = F.one_hot(x_batch.flatten(), num_classes=num_vals)

                x_reconstructed[:, recon_idx:recon_idx + num_vals] = x_batch

                # Updating indices
                recon_idx += num_vals
                output_idx += num_vals

            elif self.var_info[var]['dtype'] == 'numerical':
                # b = prob_d.shape[0] # batch size
                # m = prob_d.shape[1] # output dimension
                # below is unnecessary because already performed in self.decode()
                # prob_d = prob_d.view(prob_d.shape[0], -1, self.num_vals) # -1 is inferred from other dims (what is left)
                # p = prob_d.view(-1, self.total_num_vals) # merging batch and output dims (lists of possible outputs)
                # we want one sample per number of possible values (e.g. pixel values)

                mu, log_var = torch.chunk(params[:, output_idx:output_idx + num_vals], 2, dim=1)
                '''
                if self.scale == 'standardize':
                    mu_orig, std_orig = self.var_info[var]['standardize']
                    mu, log_var = destand_num_dist(mu_orig, std_orig, mu, log_var)
                elif self.scale == 'normalize':
                    min, max = self.var_info[var]['normalize']
                    mu, log_var = denorm_num_dist(min, max, mu, log_var)
                elif self.scale == 'none':
                    pass
                '''

                # Extracting mu and std. values.
                # mu = torch.tensor([output[:, var_index]]) # The mu's extracted from the output
                std = torch.exp(0.5 * log_var)
                # std = torch.exp(0.5*output[:, var_index+1]) # The sigma's extracred from the output
                x_batch = torch.normal(mu, std)  # .view(b, m)
                x_reconstructed[:, recon_idx:recon_idx + 1] = x_batch

                # Updating indices
                recon_idx += 1
                output_idx += num_vals

            # elif self.distribution == 'bernoulli':
            #     x_new = torch.bernoulli(prob_d) # output dim is already (batch, output dims, 1)
            else:
                raise ValueError('Either `gaussian`, `categorical`, or `bernoulli`')
        return x_reconstructed

    def log_prob(self, x, params):
        # calculating the log−probability which is later used for ELBO
        #prob_d = self.decode(z)  # probability output or real if naturals
        params = params.to(self.device)
        log_p = torch.zeros((params.shape[0], len(self.var_info)))
        params_idx = 0
        x_idx = 0
        for _, var in enumerate(self.var_info):
            if self.var_info[var]['dtype'] == 'categorical':
                num_vals = self.var_info[var]['num_vals']

                if self.natural:  # note that outputs are just logits of probability
                    probs = self.softmax(params[:, params_idx:params_idx + num_vals])
                else:
                    probs = params[:, params_idx:params_idx + num_vals]

                log_p[:, var] = log_categorical(x[:, x_idx:x_idx + num_vals], probs, reduction='sum', dim=1)  # .sum(-1)

                params_idx += num_vals
                x_idx += num_vals

            elif self.var_info[var]['dtype'] == 'numerical':  # Gaussian
                num_vals = self.var_info[var]['num_vals']  # always 2

                if self.natural:
                    # -*softplus has been applied to softplus
                    eta1, eta2 = torch.chunk(params[:, params_idx:params_idx + num_vals], 2, dim=1)
                    # restricting eta2 to be -inf < eta2 < 0
                    mu, log_var = -0.5 * eta1 / eta2, torch.log(-0.5 / eta2)
                else:
                    mu, log_var = torch.chunk(params[:, params_idx:params_idx + num_vals], 2, dim=1)

                '''
                # denormalizing
                #mu = torch.tanh(mu)
                
                if self.scale == 'standardize':
                    mu_orig, std_orig = self.var_info[var]['standardize']
                    mu, log_var = destand_num_dist(mu_orig, std_orig, mu, log_var)
                elif self.scale == 'normalize':
                    min, max = self.var_info[var]['normalize']
                    mu, log_var = denorm_num_dist(min, max, mu, log_var)
                elif self.scale == 'none':
                    pass
                '''

                log_p[:, var] = log_normal(x[:, x_idx:x_idx + 1], mu, log_var, reduction='sum', dim=1)
                params_idx += num_vals
                x_idx += 1

            else:
                raise ValueError('Either `gaussian`, `categorical`')
        # summing log_p for each variable (i.e. we have on log prob per batch)
        #   - e.g. all categories in categorical, i.e. dimension is batch
        return torch.sum(log_p, axis=1).to(self.device)

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

class VampPrior(nn.Module):
    def __init__(self, L, D, num_vals, encoder, num_components, data=None):
        super(VampPrior, self).__init__()

        self.L = L
        self.D = D
        self.num_vals = num_vals

        self.encoder = encoder

        # pseudoinputs
        u = torch.rand(num_components, D) * self.num_vals
        self.u = nn.Parameter(u)

        # mixing weights
        self.w = nn.Parameter(torch.zeros(self.u.shape[0], 1, 1)) # K x 1 x 1

    def get_params(self):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.encoder.encode(self.u) #(K x L), (K x L)
        return mean_vampprior, logvar_vampprior

    def sample(self, batch_size):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0) # K x 1 x 1 
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])
            else:
                z = torch.cat((z, mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])), 0)
        return z

    def log_prob(self, z):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params() # (K x L) & (K x L)

        # mixing probabilities
        w = F.softmax(self.w, dim=0) # K x 1 x 1

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        mean_vampprior = mean_vampprior.unsqueeze(1) # K x 1 x L
        logvar_vampprior = logvar_vampprior.unsqueeze(1) # K x 1 x L

        log_p = log_normal_diag(z, mean_vampprior, logvar_vampprior) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob # B 

def stand_num(var_info, data):
    # whole batch
    # i.e. z-score scaling
    new_data = data.clone()
    idx = 0
    for x_idx, var in enumerate(var_info):
        num_vals = var_info[var]['num_vals']
        if var_info[var]['dtype'] == 'numerical':
            mean, std = var_info[var]['standardize']
            new_data[:, idx] = (data[:, idx] - mean) / std
            idx += 1
        else:
            idx += num_vals
        # ignoring categorical
    return new_data


def destand_num_params(var_info, data):
    # whole batch
    new_data = data.clone()
    idx = 0
    for x_idx, var in enumerate(var_info):
        num_vals = var_info[var]['num_vals']
        if var_info[var]['dtype'] == 'numerical':
            mean, std = var_info[var]['normalize']
            new_data[:, idx:idx+1] = (data[:, idx:idx+1] * std) + mean
        # ignoring categorical
        idx += num_vals
    return new_data



def destand_num_dist(mu_orig, std_orig, mu, log_var):
    # single params
    new_mu = mu * std_orig + mu_orig
    std = torch.sqrt(torch.exp(log_var))
    new_log_var = torch.log((std * std_orig) ** 2)
    return new_mu, new_log_var



def norm_num(var_info, data):
    # whole batch
    # i.e. min max scaling
    new_data = data.clone()
    idx = 0
    for x_idx, var in enumerate(var_info):
        num_vals = var_info[var]['num_vals']
        if var_info[var]['dtype'] == 'numerical':
            min, max = var_info[var]['normalize']
            new_data[:, idx] = (data[:, idx] - min) / (max-min)
            idx += 1
        else:
            idx += num_vals
        # ignoring categorical
    return new_data


def denorm_num_params(var_info, data):
    # whole batch
    new_data = data.clone()
    idx = 0
    for x_idx, var in enumerate(var_info):
        num_vals = var_info[var]['num_vals']
        if var_info[var]['dtype'] == 'numerical':
            min, max = var_info[var]['normalize']
            new_data[:, idx:idx+1] = (data[:, idx:idx+1] * (max-min)) + min
            log_var = data[:, idx+1:idx+2]
            std = torch.sqrt(torch.exp(log_var))
            new_log_var = torch.log((std * (max-min)) ** 2)
            new_data[:, idx+1:idx+2] = new_log_var
        # ignoring categorical
        idx += num_vals
    return new_data


def denorm_num_dist(min, max, mu, log_var):
    # single params
    new_mu = mu * (max-min) + min
    std = torch.sqrt(torch.exp(log_var))
    new_log_var = torch.log((std * (max-min)) ** 2)
    return new_mu, new_log_var


class VAE(nn.Module):
    def __init__(self, total_num_vals, L, var_info, D, M, natural, scale, device, prior:str,beta=1.0):
        super().__init__()

        encoder_net = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                                    nn.Linear(M, M), nn.LeakyReLU(),
                                    nn.Linear(M, 2 * L))

        # Goes from latent space back to feature-dimension
        decoder_net = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
                                    nn.Linear(M, M), nn.LeakyReLU(),
                                    nn.Linear(M, total_num_vals))

        encoder_net.to(device)
        decoder_net.to(device)
        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(var_info=var_info, decoder_net=decoder_net, total_num_vals=total_num_vals,
                               natural=natural, scale=scale, device=device)
        if prior == 'vampPrior':
            self.prior = VampPrior(L=L, D=D, num_vals=total_num_vals, encoder=self.encoder, num_components=len(var_info))
        else:
            self.prior = Prior(L=L)
        self.total_num_vals = total_num_vals
        self.var_info = var_info  # contains type of likelihood for variables
        self.D = D
        self.device = device
        self.scale = scale
        self.beta = beta
        # todo: self.normalizing stuff

    def forward(self, x, loss=True, reconstruct=False, nll=False, reduction='sum'):

        # Initializing outputs
        RECONSTRUCTION = None
        LOSS = None
        NLL = None
        '''
        # batch normalization
        reference_idx = 0
        for idx in self.var_info.keys():
            if self.var_info[idx]['dtype'] == 'numerical':
                mean = torch.mean(x[:, reference_idx])
                std = torch.std(x[:, reference_idx])
                self.var_info[idx]['normalize'] = (mean, std)
                reference_idx += 1
            else:
                reference_idx += self.var_info[idx]['num_vals']
        '''
        if self.scale == 'standardize': # Mean 0, Std 1
            x = stand_num(self.var_info, x)
        elif self.scale == 'normalize': # Between 0 and 1
            x = norm_num(self.var_info, x)
        elif self.scale == 'none':
            pass

        # Encode
        x = x.to(self.device) # Now normalized
        mu_e, log_var_e = self.encoder.encode(x)
        # sample in latent space
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e) # Sampling latent z from learned mu and log var.
        z = z.to(self.device)

        params = self.decoder.decode(z)  # probability output - 

        if self.scale == 'standardize': # Mean 0, Std 1
            params = destand_num_params(self.var_info, params)
        elif self.scale == 'normalize': # Between 0 and 1
            params = denorm_num_params(self.var_info, params)
        elif self.scale == 'none':
            pass

        # reconstruct
        if reconstruct:
            # Sample/predict
            RECONSTRUCTION = self.decoder.sample(params)
            # updated

        if loss:
            # ELBO
            # reconstruction error
            #if self.scale == 'standardize':
            #    RE = nn.MSELoss()(x,z)
            RE = self.decoder.log_prob(x, params)  # z is decoded back
            # Kullback–Leibler divergence, regularizer
            # todo mean or sum?
            # torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            #KL = torch.mean(-0.5 * torch.sum(1 + log_var_e - mu_e ** 2 - log_var_e.exp(), dim = 1), dim = 0)
            KL = torch.sum((self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)),axis=1)
            # summing the loss for this batch
            # torch.sqrt(F.mse_loss(self.decoder.sample(z), x))
            LOSS = -(RE + self.beta * KL).sum() # self.calculate_RMSE(x, self.decoder.sample(params))['regular'] 


        if nll:
            assert (nll and loss) == True, 'loss also has to be true in input for forward call'
            # loss
            # first find NLL averaged over variables in data
            # then mean the batch
            # updated
            NLL = (-RE / self.D).mean().detach()

        return RECONSTRUCTION, LOSS, NLL

        
    def calculate_RMSE(self, x, x_recon):
        var_idx = 0
        MSE = {'regular': torch.tensor(0.), 'numerical': torch.tensor(0.), 'categorical': torch.tensor(0.)}  # Initializing RMSE score
        RMSE = {}

        # Number of variable-types
        D = len(self.var_info.keys())
        num_numerical = sum([self.var_info[var]['dtype'] == 'numerical' for var in self.var_info.keys()])
        num_categorical = D - num_numerical
        
        # Num observations in batch
        obs_in_batch = x.shape[0] 
        for var in self.var_info.keys():
            num_vals = self.var_info[var]['num_vals']

            # Getting length of slice
            if self.var_info[var]['dtype'] == 'numerical':
                idx_slice = 1
            else:  # categorical
                idx_slice = num_vals

            # Imputation targets and predictions - for variable
            var_targets = x[:, var_idx:var_idx + idx_slice]
            var_preds = x_recon[:, var_idx:var_idx + idx_slice]

            # MSE per variable
            assert F.mse_loss(var_preds, var_targets)*idx_slice == torch.sum((var_targets - var_preds) ** 2) / obs_in_batch
            MSE_var = torch.sum((var_targets - var_preds) ** 2) / obs_in_batch

            # Summing variable MSEs - (outer-most sum of formula)
            #MSE += MSE_var
            # Summing variable MSEs - (outer-most sum of formula)
            MSE['regular'] += MSE_var
            # Also adding to variable type MSE
            MSE[self.var_info[var]['dtype']] += MSE_var

            # Updating current variable index
            if self.var_info[var]['dtype'] == 'numerical':
                var_idx += 1
            else:  # categorical
                var_idx += num_vals

        # Taking square-root (RMSE), and averaging over features. (As seen in formula)
        for (dtype, type_count) in {'regular': D, 'numerical': num_numerical, 'categorical': num_categorical}.items():
            RMSE[dtype] = torch.sqrt(MSE[dtype]) / type_count

        # Updating numerical and categorical RMSE to represent an accurate ratio of Regular RMSE - that sums to regular. 
        RMSE['numerical'], RMSE['categorical'] = [RMSE['regular'] * (RMSE[dtype] / (RMSE['numerical'] + RMSE['categorical'])) for dtype in ['numerical', 'categorical']]    

        return RMSE


class Baseline():
    """
    Imputes the mean of each continuous variable, and the mode of the discrete (categorical) variables.


    model = Baseline()
    model.train(train_data)
    model.evaluate(test_data, class_idxs = train_data.data_dict['class_idxs'])


    """

    def __init__(self, var_info, train_loader, test_loader):

        self.var_info = var_info
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.predictions = {}
        self.ids = {"categorical": [[i, var['name']] for i, var in enumerate(self.var_info.values()) if
                                    var['dtype'] == 'categorical'],
                    "numerical": [[i, var['name']] for i, var in enumerate(self.var_info.values()) if
                                  var['dtype'] == 'numerical']}
        self.log_p = None
        self.MSE = {}  # MSE's per numerical variable
        self.accuracy = {}  # Accuracies per categorical variable

    def train(self):

        # Concatenating batches
        data = []
        for indx_batch, batch in enumerate(self.train_loader):
            data.append(batch)
        data = torch.cat(data)

        predictions = {'categorical': {}, 'numerical': {}}

        # Returns class of highest count per attribute
        for attr, _ in self.ids['categorical']:
            predictions['categorical'][attr] = np.argmax(np.unique(data[:, attr], return_counts=True)[1])

        # Avg. prediction per attribute
        for attr, _ in self.ids['numerical']:
            # mean and variance
            predictions['numerical'][attr] = (data[:, attr].mean(), data[:, attr].var())

        self.predictions = predictions

    def evaluate(self):

        # Concatenating batches
        data = []
        for indx_batch, batch in enumerate(self.test_loader):
            data.append(batch)
        data = torch.cat(data)

        self.log_p = 0

        # Calculating and summing log probabilities per
        for attr, _ in self.ids['categorical']:
            num_classes = self.var_info[attr]['num_vals']
            # Actual class_id (1 X batch_size)
            x = data[:, attr]

            # predicted class_id
            predicted_class = self.predictions['categorical'][attr]

            # Calculating accuracy for categorical variables
            self.accuracy[attr] = np.mean([obs == predicted_class for obs in x])

            # One-hotting entire column based on predicted class
            onehot = np.zeros((data.shape[0], num_classes))
            onehot[:, predicted_class] = 1

            prob_d = torch.Tensor(onehot)

            self.log_p += log_categorical(x, prob_d, num_classes=num_classes, reduction='sum', dim=-1).sum(-1)

        for attr, _ in self.ids['numerical']:
            x = data[:, attr]

            # Calculating MSE for numerical variables
            mu, log_var = self.predictions['numerical'][attr]
            self.MSE[attr] = ((x - mu) ** 2).mean()

            # TODO: Should be exchanged for a normal (per variable (in prob dist))
            log_normal(data, mu, log_var, reduction=None, dim=-1).sum(-1)

    def plot_results(self, plotting=True):

        if plotting:
            plt.rcParams["figure.figsize"] = (10, 4)
            # plotting
            fig, [ax1, ax2] = plt.subplots(2, 1)
            fig.subplots_adjust(hspace=0.6)

            ax1.bar([attr_name for (attr, attr_name) in self.ids['numerical']], self.MSE.values())
            ax1.title.set_text('MSE for numerical variables - Baseline')

            ax2.bar([attr_name for (attr, attr_name) in self.ids['categorical']], self.accuracy.values())
            ax2.title.set_text('Acc. for categorical variables - Baseline')
            fig.show()

        # Numerical data
        # numerical = test_data.data_dict['numerical']
        # mu var af træningsdata
        # l















