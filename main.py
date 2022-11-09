import argparse
from logger import Logger
import numpy as np
import torch
from train import training
from models import *
from dataloaders import Boston


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

    # general
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    # model parameters
    parser.add_argument('--lr', help='Starting learning rate',default=3e-4, type=float)
    parser.add_argument('--max_epochs', help='"Number of epochs to train for"', default=1000, type=int)
    parser.add_argument('--max_patience', help='"If training does not improve for longer than --max_patience epochs, it is stopped"', default=20, type=int)

    # tensorboard
    parser.add_argument('--log_dir', help='Store logs in this directory during training.',
                        default='runs', type=str)
    parser.add_argument('--write', help='Saves the training logs', dest='write',
                        action='store_true')
    
    args = parser.parse_args()

    logger = Logger(directory=args.log_dir, comment="_HI_VAE", write=args.write)

    D = 64  # input dimension
    L = 16  # number of latents
    M = 256  # the number of neurons in scale (s) and translation (t) nets

    likelihood_type = 'categorical'

    if likelihood_type == 'categorical':
        num_vals = 1
    elif likelihood_type == 'bernoulli':
        num_vals = 1

    train_data = Boston(mode='train')
    val_data = Boston(mode='val')
    test_data = Boston(mode='test')

    training_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)


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



