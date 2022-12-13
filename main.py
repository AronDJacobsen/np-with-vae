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
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])

    # model parameters
    parser.add_argument('--lr', help='Starting learning rate',default=3e-4, type=float)
    parser.add_argument('--batch_size', help='"Batch size"', default=32, type=int)
    parser.add_argument('--natural', dest='natural', action='store_true')

    parser.add_argument('--max_epochs', help='"Number of epochs to train for"', default=1000, type=int)

    parser.add_argument('--train', dest='is_train', action='store_true')

    parser.add_argument('--max_patience', help='"If training does not improve for longer than --max_patience epochs, it is stopped"', default=20, type=int)

    # tensorboard
    parser.add_argument('--log_dir', help='Store logs in this directory during training.',
                        default='runs', type=str)
    parser.add_argument('--write', help='Saves the training logs', dest='write',
                        action='store_true')
    # dataset
    parser.add_argument('--dataset', type=str, default='avocado', choices=['boston', 'avocado', 'energy', 'bank'])

    args = parser.parse_args()
    logger = Logger(directory=args.log_dir, comment="_VAE", write=args.write)

    ## Creating directory for test results
    result_dir = 'results/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)
    name = 'vae'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running {}".format(device))

    if device == 'cpu':
        pin_memory = False
    else:
        pin_memory = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # Loading dataset
    # information about variables and dataset loaders
    output = load_dataset(dataset_name=args.dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed, pin_memory=pin_memory)
    # extracting output
    info, loaders = output
    (var_info, var_dtype) = info
    train_loader, val_loader, test_loader = loaders

    # finding values for encoder and decoder networks
    L = len(var_info.keys())  # number of latents, i.e. z, (later 2x, i.e. mu and sigma per variable but then sample z)
    # total of the number of values per variable (i.e. output dim of decoder)
    total_num_vals = 0
    D = 0 # total number of variables in data
    for var in var_info.keys():
        total_num_vals += var_info[var]['num_vals']
        # categorical input dimension is one-hot thus num_vals is input
        if var in var_dtype['categorical']: # todo and num_vals
            D += var_info[var]['num_vals']
        # numerical just har input dim of 1
        else:
            D += 1

    # training, evaluating and plotting results for baseline. 
    # print('Running Baseline')
    # baseline = Baseline(var_info, train_loader, test_loader)
    # baseline.train()
    # baseline.evaluate()
    # baseline.plot_results(plotting = True)


    # TODO: make hparam
    M = 256  # the number of neurons in scale (s) and translation (t) nets, i.e. hidden dimension in decoder/encoder


    prior = torch.distributions.MultivariateNormal(torch.zeros(L), torch.eye(L))
    model = VAE(total_num_vals=total_num_vals, L=L, var_info = var_info, D=D, M=M,natural=args.natural, device=device)
    model = model.to(device)
    # OPTIMIZER
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=args.lr)
    print(f'training = {args.is_train}')
    # Training procedure
    if args.is_train:
        nll_val = training(name=logger.dir, max_patience=args.max_patience, num_epochs=args.max_epochs, model=model,
                       optimizer=optimizer,
                       train_loader=train_loader, val_loader=val_loader,var_info=var_info,natural=args.natural, device=device)

        print(nll_val)
    nll_val = [0]

    # Save and plot test_results
    get_test_results(nll_val=nll_val, result_path = result_dir + name, test_loader=test_loader,var_info=var_info, D=D, natural=args.natural, device=device)

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



