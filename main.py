import argparse
from logger import Logger
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('-seed', type=int, default=None)

    # tensorboard
    parser.add_argument('-log_dir', help='Store logs in this directory during training.',
                        default='runs', type=str)
    parser.add_argument('-write', help='Saves the training logs', dest='write',
                        action='store_true')
    
    args = parser.parse_args()

    logger = Logger(directory=args.log_dir, comment="_HI_VAE", write=args.write)
    # name = "Train"
    # for j in range(10):
    #     scalars = {}
    #     t = np.linspace(0, np.pi, 7)
    #     y = lambda t: np.cos(t)+j
    #     for i in t:
    #         scalars[i] = y(i)
    #
    #     logger.write_to_board(name=name, scalars=scalars, index=j)



