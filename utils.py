import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from models import *


def get_model(model_name, total_num_vals, L, var_info, D, M, natural, device):
    if model_name == 'VAE':
        return VAE(total_num_vals=total_num_vals, L=L, var_info=var_info, D=D, M=M, natural=natural, device=device)
    elif model_name == 'BASELINE':
        # todo make it train and able to evaluate like the other
        return Baseline()
    else:
        raise NotImplementedError("Specified model is currently not implemented.")


def load_model(model_path, model, device='cuda'):
    state_dict = torch.load(model_path + 'best.ckpt', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# def evaluation(test_loader, var_info, model, model_best=None, epoch=None, M=256,natural=False,device=None):
def evaluation(model, data_loader, device, reduction='sum'):
    # EVALUATION
    '''
    if model_best is None:
        D = D
        L = len(var_info.keys())
        total_num_vals = 0
        for var in var_info.keys():
            total_num_vals += var_info[var]['num_vals']
            # categorical input dimension is one-hot thus num_vals is input
        model_best = VAE(total_num_vals=total_num_vals, L=L, var_info = var_info, D=D, M=M, natural=natural, device=device)
        model_best.to(device)
        # load best performing model
        model_best.load_state_dict(torch.load(model_path + model_name + '.model'))

    model_best.eval()
    '''
    # model = model.eval()
    total_loss = 0.
    N = 0.
    performance_df = pd.DataFrame()
    for indx_batch, batch in enumerate(data_loader):
        batch = batch.to(device)
        # test_batch = torch.stack(test_batch[1]).float() # TODO: To access only one attribute - only needed as long as no multi-head
        # TODO: adjust to normal batch
        # numerical = test_batch[0].float()
        # categorical = test_batch[1]

        # concatenate into big input
        # test_batch = torch.cat((numerical, categorical), dim=1)

        # test_batch = test_batch[1]
        # TODO: this was also implemented in train.training and utils.samples_generated
        output, loss, performance = model.forward(batch, reduction=reduction)
        performance_df = pd.concat([performance_df, pd.DataFrame.from_dict([performance])])

        total_loss = total_loss + loss['loss']
        N = N + batch.shape[0]

    loss = total_loss / N

    # if epoch is None:
    #    print(f'FINAL LOSS: nll={loss}')
    # else:
    #    print(f'Epoch: {epoch}, val nll={loss}')

    return loss, performance_df


# def evaluate_to_table(test_loader, var_info, name=None, model_best=None, epoch=None, M=256,natural=False,device=None):
def evaluate_to_table(model, data_loader, device):
    # EVALUATION
    '''
    if model_best is None:
        D = len(var_info.keys())
        L = D
        total_num_vals = 0
        for var in var_info.keys():
            total_num_vals += var_info[var]['num_vals']
        model_best = VAE(total_num_vals=total_num_vals, L=L, var_info = var_info, D=D, M=M, natural=natural, device=device)
        model_best.to(device)
        # load best performing model
        model_best.load_state_dict(torch.load(name+'.model'))

    model_best.eval()
    '''

    loss = 0.
    N = 0.
    for indx_batch, batch in enumerate(data_loader):
        batch = batch.to(device)
        loss_t = model.forward(batch, reduction='sum')['loss']
        loss = loss + loss_t.item()
        N = N + batch.shape[0]
    loss = loss / N

    '''
    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')
    '''

    return loss  # + mse? (NLL, MSE)


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    # TODO: same deal as in samples_generated, evaluation and train.training
    x = torch.stack(next(iter(test_loader))[1]).float().detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(save_path, name, data_loader, extra_name=''):
    # TODO: originally:
    # x = next(iter(data_loader)).detach().numpy()
    # x = torch.stack(next(iter(data_loader))[1]).float().detach().numpy()
    x = next(iter(data_loader))[1]
    # To access only one (categorical) attribute - only needed as long as no multi-head
    # Also done in train.training and utils.evaluation

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(save_path + name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()


def plot_curve(name):  # , nll_val):
    # plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


# def get_test_results(nll_val, result_path, test_loader,var_info, D=256, natural=None, device=None):
#    test_loss = evaluation(test_loader, var_info, name=result_path, D=D, natural=natural,device=device)
#    f = open(result_path + '_test_loss.txt', "w")

def get_test_results(model, result_path, model_name, test_loader, var_info, device):
    # loading best model

    model = load_model(model_path=result_path, model=model, device=device)
    test_loss, performance_df = evaluation(model, test_loader, device)
    f = open(result_path + 'test_loss.txt', "w")
    f.write(str(test_loss))
    f.close()

    # samples_real(result_path, test_loader)

    plot_curve(result_path)  # , nll_val)

    imputation_error = imputation_score(test_loader, var_info, model, name=result_path, device=device,
                                        imputation_ratio=0.5)


def imputation_score(test_loader, var_info, model, name=None, device=None, imputation_ratio=0.5):
    model.eval()
    RMSE = 0
    D = len(var_info.keys())

    for indx_batch, test_batch in enumerate(test_loader):
        imputation_mask = np.ones(test_batch.shape)
        for i, observation in enumerate(test_batch):

            # Random draw of variables to set zero - based on imputation ratio
            imputation_variables = np.random.choice(list(var_info.keys()), size=int(len(var_info) * imputation_ratio),
                                                    replace=False)
            imp_idx = 0  # total number of variables in data
            for var in range(len(list(var_info.keys()))):
                # Get indeces to impute
                if var_info[var]['dtype'] == 'categorical':
                    idx1 = imp_idx
                    imp_idx += var_info[var]['num_vals']
                    idx2 = imp_idx
                else:
                    idx1 = imp_idx
                    imp_idx += 1
                    idx2 = imp_idx

                if var in imputation_variables:
                    # Put into mask
                    imputation_mask[i, :][idx1:idx2] = 0

        imputed_test_batch = (test_batch * imputation_mask)

        # for var in imputation_variables:
        # OBS der skal også gemmes hvilke elementer der blev sat til 0, så de senere kan sammenlignes

        #    idx_range = (sum([var_info[i]['num_vals'] for i in range(var)]), var_info[var]['num_vals'])
        #    test_batch[observation,:][idx_range[0]:idx_range[0]+idx_range[1]]

        imputed_test_batch = imputed_test_batch.to(device)

        # with torch no grad
        batch_idx = 0
        reconstructed_test_batch = model.forward(imputed_test_batch)['output'].detach().numpy()

        # The following is the reconstructed values of the imputed scores:
        # reconstructed_test_batch[imputation_mask == 0]

        # Actual imputed values
        var_idx = 0
        MSE = 0
        for var in var_info.keys():
            num_vals = var_info[var]['num_vals']

            # Imputation targets and predictions.
            imputation_targets = test_batch[:, var_idx:var_idx + 1][imputation_mask[:, var_idx:var_idx + 1] == 0]
            imputation_preds = reconstructed_test_batch[:, var_idx:var_idx + 1][
                imputation_mask[:, var_idx:var_idx + 1] == 0]

            MSE += torch.mean((imputation_targets - imputation_preds) ** 2)

            if var_info[var]['dtype'] == 'numerical':
                var_idx += 1
            else:  # categorical
                var_idx += num_vals

        RMSE += torch.sqrt(MSE) / D

    print(RMSE / indx_batch)

    # output = model_best.forward(imputed_test_batch, reduction='avg')['output']
    # loss = loss + loss_t.item()
    # N = N + test_batch.shape[0]
    # loss = loss / N
    return None
