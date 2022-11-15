import argparse
from logger import Logger
import numpy as np
import torch
from train import training
from models import *
from dataloaders import Boston
from utils import samples_real, plot_curve, evaluation, get_test_results
from torch.utils.data import Dataset, DataLoader, random_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

    # general
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    # model parameters
    parser.add_argument('--lr', help='Starting learning rate',default=3e-4, type=float)
    parser.add_argument('--batch_size', help='"Batch size"', default=32, type=int)

    parser.add_argument('--max_epochs', help='"Number of epochs to train for"', default=1000, type=int)
    parser.add_argument('--max_patience', help='"If training does not improve for longer than --max_patience epochs, it is stopped"', default=20, type=int)

    # tensorboard
    parser.add_argument('--log_dir', help='Store logs in this directory during training.',
                        default='runs', type=str)
    parser.add_argument('--write', help='Saves the training logs', dest='write',
                        action='store_true')

    # TODO: input argument as batch size

    args = parser.parse_args()

    logger = Logger(directory=args.log_dir, comment="_HI_VAE", write=args.write)

    D = 1  # input dimension, e.g. image dimensions
    L = 16  # number of latents
    M = 256  # the number of neurons in scale (s) and translation (t) nets

    # likelihood_type = 'categorical'
    likelihood_type = 'gaussian'
    num_vals = 2
    D = 13 # number of numerical attributes

    if likelihood_type == 'categorical':
        num_vals = 2 # Should be equal to number of classes -> i.e. dependent on dataset and attribute
    elif likelihood_type == 'bernoulli':
        num_vals = 1

    # TODO: implement random split based on seed
    train_data = Boston(mode='train')
    val_data = Boston(mode='val')
    test_data = Boston(mode='test')

    # TODO: Should batch_size == D? -> only works like so
    training_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=True) # drop_last to drop incomplete batches
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True) # drop_last to drop incomplete batches
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True) # drop_last to drop incomplete batches

    ## Creating directory for test results
    result_dir = 'results/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)
    name = 'vae'

    encoder = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * L))

    decoder = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, num_vals * D))


    prior = torch.distributions.MultivariateNormal(torch.zeros(L), torch.eye(L))
    model = VAE(encoder_net=encoder, decoder_net=decoder, num_vals=num_vals, L=L, likelihood_type=likelihood_type)

    # OPTIMIZER
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=args.lr)

    # Training procedure
    nll_val = training(name=logger.dir, max_patience=args.max_patience, num_epochs=args.max_epochs, model=model,
                       optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader)

    print(nll_val)

    # Save and plot test_results
    get_test_results(nll_val=nll_val, result_path = result_dir + name, test_loader=test_loader)

    # ### testing ###
    # logger.log("Training using {}".format(args.device))
    #
    # name = "Train"
    # for j in range(10):
    #     scalars = {}
    #     t = np.linspace(0, np.pi, 7)
    #     y = lambda t: np.cos(t)+j
    #     for i in t:
    #         scalars[i] = y(i)
    #
    #     logger.write_to_board(name=name, scalars=scalars, index=j)



