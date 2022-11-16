import argparse
from logger import Logger
import numpy as np
import torch
from train import training
from models import *
from dataloaders import *
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

    parser.add_argument('--max_epochs', help='"Number of epochs to train for"', default=200, type=int)
    parser.add_argument('--max_patience', help='"If training does not improve for longer than --max_patience epochs, it is stopped"', default=20, type=int)

    # tensorboard
    parser.add_argument('--log_dir', help='Store logs in this directory during training.',
                        default='runs', type=str)
    parser.add_argument('--write', help='Saves the training logs', dest='write',
                        action='store_true')
    # dataset
    parser.add_argument('--dataset', type=str, default='boston', choices=['boston', 'avocado', 'energy', 'bank'])

    args = parser.parse_args()

    logger = Logger(directory=args.log_dir, comment="_VAE", write=args.write)

    ## Creating directory for test results
    result_dir = 'results/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)
    name = 'vae'


    # TODO: implement random split based on seed
    #train_data = Boston(mode='train')
    #val_data = Boston(mode='val')
    #test_data = Boston(mode='test')

    # TODO: Should batch_size == D? -> only works like so

    #training_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=True) # drop_last to drop incomplete batches
    #val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True) # drop_last to drop incomplete batches
    #test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True) # drop_last to drop incomplete batches

    # Loading dataset
    # information about variables and dataset loaders
    output = load_dataset(dataset_name=args.dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    # extracting output
    var_info, loaders = output
    train_loader, val_loader, test_loader = loaders

    # finding values for encoder and decoder networks
    D = len(var_info.keys()) # total number of variables in data
    L = D  # number of latents (later 2x, i.e. mu and sigma per variable, i.e. output dim of encoder)
    # total of the number of values per variable (i.e. output dim of decoder)
    total_num_vals = 0
    for var in var_info.keys():
        total_num_vals += var_info[var]['num_vals']

    M = 256  # the number of neurons in scale (s) and translation (t) nets, i.e. hidden dimension in decoder/encoder


    # TODO: do this in the model
    # creating encoder and decoder network
    encoder = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, 2 * L))

    decoder = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
                            nn.Linear(M, M), nn.LeakyReLU(),
                            nn.Linear(M, total_num_vals))


    prior = torch.distributions.MultivariateNormal(torch.zeros(L), torch.eye(L))
    model = VAE(encoder_net=encoder, decoder_net=decoder, num_vals=total_num_vals, L=L, var_info = var_info)

    # OPTIMIZER
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=args.lr)

    # Training procedure
    nll_val = training(name=logger.dir, max_patience=args.max_patience, num_epochs=args.max_epochs, model=model,
                       optimizer=optimizer,
                       train_loader=train_loader, val_loader=val_loader)

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



