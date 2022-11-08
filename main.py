import argparse
from logger import Logger
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')

    # general
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    # model parameters


    # tensorboard
    parser.add_argument('--log_dir', help='Store logs in this directory during training.',
                        default='runs', type=str)
    parser.add_argument('--write', help='Saves the training logs', dest='write',
                        action='store_true')
    
    args = parser.parse_args()

    logger = Logger(directory=args.log_dir, comment="_HI_VAE", write=args.write)

    ### load data
    train_data = Boston(mode='train')
    val_data = Boston(mode='val')
    test_data = Boston(mode='test')

    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # create dummy dataloader to get a batch of 1
    tester_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    
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



