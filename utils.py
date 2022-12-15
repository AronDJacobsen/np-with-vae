import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from models import *


def get_model(model_name, total_num_vals, L, var_info, D, M, natural, device,prior,beta, scale, scale_type, decay):
    if model_name == 'VAE':
        return VAE(total_num_vals=total_num_vals, L=L, var_info=var_info, D=D, M=M, natural=natural, device=device,prior=prior,beta=beta, scale=scale, scale_type=scale_type, decay=decay)
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
    #performance_df = pd.DataFrame()
    for indx_batch, batch in enumerate(data_loader):
        batch = batch.to(device)
        _, loss, _ = model.forward(batch, reduction=reduction)

        total_loss = total_loss + loss
        N = N + batch.shape[0]

    loss = total_loss / N

    # if epoch is None:
    #    print(f'FINAL LOSS: nll={loss}')
    # else:
    #    print(f'Epoch: {epoch}, val nll={loss}')

    return loss #, performance_df




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

def calculate_RMSE(var_info, x, x_recon):
    var_idx = 0
    MSE = {'regular': 0, 'numerical': 0, 'categorical': 0}  # Initializing RMSE score
    RMSE = {}

    # Number of variable-types
    D = len(var_info.keys())
    num_numerical = sum([var_info[var]['dtype'] == 'numerical' for var in var_info.keys()])
    num_categorical = D - num_numerical
    
    # Num observations in batch
    obs_in_batch = x.shape[0] 
    for var in var_info.keys():
        num_vals = var_info[var]['num_vals']

        # Getting length of slice
        if var_info[var]['dtype'] == 'numerical':
            idx_slice = 1
        else:  # categorical
            idx_slice = num_vals

        # Imputation targets and predictions - for variable
        var_targets = x[:, var_idx:var_idx + idx_slice]
        var_preds = x_recon[:, var_idx:var_idx + idx_slice]

        # MSE per variable
        MSE_var = torch.sum((var_targets - var_preds) ** 2) / obs_in_batch

        # Summing variable MSEs - (outer-most sum of formula)
        #MSE += MSE_var
        # Summing variable MSEs - (outer-most sum of formula)
        MSE['regular'] += MSE_var
        # Also adding to variable type MSE
        MSE[var_info[var]['dtype']] += MSE_var
        # Updating current variable index
        if var_info[var]['dtype'] == 'numerical':
            var_idx += 1
        else:  # categorical
            var_idx += num_vals

    # Taking square-root (RMSE), and averaging over features. (As seen in formula)
    for (dtype, type_count) in {'regular': D, 'numerical': num_numerical, 'categorical': num_categorical}.items():
        RMSE[dtype] = torch.sqrt(MSE[dtype]).item() / type_count

    # Updating numerical and categorical RMSE to represent an accurate ratio of Regular RMSE - that sums to regular. 
    RMSE['numerical'], RMSE['categorical'] = [RMSE['regular'] * (RMSE[dtype] / (RMSE['numerical'] + RMSE['categorical'])) for dtype in ['numerical', 'categorical']]    

    return RMSE

def calculate_imputation_error(var_info, test_batch, model, device, imputation_ratio):

    # Number of variable-types
    D = len(var_info.keys())
    num_numerical = sum([var_info[var]['dtype'] == 'numerical' for var in var_info.keys()])
    num_categorical = D - num_numerical

    imputation_RMSE = {} # Initializing RMSE score

    # Getting imputation mask
    imputation_mask = create_imputation_mask(test_batch, var_info, imputation_ratio = 0.5)

    # Setting imputed variables to zero
    imputed_test_batch = (test_batch * imputation_mask)
    imputed_test_batch = imputed_test_batch.to(device)

    # Getting the reconstructed test_batch by sending the imputed test batch through VAE
    reconstructed_test_batch, _, _ = model.forward(imputed_test_batch,
                                                   reconstruct=True)  # [0]['output'].detach().numpy()

    if model.scale_type == 'outside_model':
        if model.scale == 'standardize':
            reconstructed_test_batch = destand_num(model.var_info, reconstructed_test_batch)
            test_batch = destand_num(model.var_info, test_batch)
        elif model.scale == 'normalize':
            reconstructed_test_batch = denorm_num(model.var_info, reconstructed_test_batch)
            test_batch = denorm_num(model.var_info, test_batch)

    # Calculating RMSE based on Formula in Appendix D of Ma-paper
    var_idx = 0
    imputation_MSE = {'regular': 0, 'numerical': 0, 'categorical': 0}  # Initializing RMSE score
    for var in var_info.keys():
        num_vals = var_info[var]['num_vals']

        # Getting length of slice
        if var_info[var]['dtype'] == 'numerical':
            idx_slice = 1
        else:  # categorical
            idx_slice = num_vals

        # Imputation targets and predictions - for variable
        imputation_targets = test_batch[:, var_idx:var_idx + idx_slice][imputation_mask[:, var_idx:var_idx + idx_slice] == 0]
        imputation_preds = reconstructed_test_batch[:, var_idx:var_idx + idx_slice][imputation_mask[:, var_idx:var_idx + idx_slice] == 0]

        # MSE per variable - for all unobserved slots (inner-most sum of formula)
        # The number of unobserved slots can be accessed as:
        Nd = (imputation_mask[:,
              var_idx:var_idx + 1] == 0).sum()  # We're only accessing the first slice as to not count each one-hot idx towards Nd.
        MSE_var = torch.sum((imputation_targets - imputation_preds) ** 2) / Nd

        # Summing variable MSEs - (outer-most sum of formula)
        imputation_MSE['regular'] += MSE_var
        # Also adding to variable type MSE
        imputation_MSE[var_info[var]['dtype']] += MSE_var

        # Updating current variable index
        if var_info[var]['dtype'] == 'numerical':
            var_idx += 1
        else:  # categorical
            var_idx += num_vals

    # Taking square-root (RMSE), and averaging over features. (As seen in formula)
    for (dtype, type_count) in {'regular': D, 'numerical': num_numerical, 'categorical': num_categorical}.items():
        imputation_RMSE[dtype] = torch.sqrt(imputation_MSE[dtype]).item() / type_count

    #[imputation_RMSE[dtype].append(torch.sqrt(imputation_MSE[dtype]).item() / type_count) for (dtype, type_count) in
    # {'regular': D, 'numerical': num_numerical, 'categorical': num_categorical}.items()]

    imputation_RMSE['numerical'], imputation_RMSE['categorical'] = [imputation_RMSE['regular'] * (imputation_RMSE[dtype] / (imputation_RMSE['numerical'] + imputation_RMSE['categorical'])) for dtype in ['numerical', 'categorical']]
    return imputation_RMSE


def get_test_results(model, test_loader, var_info, D, device, imputation_ratio=0.5):

    # dataframe containing all results
    results_df = pd.DataFrame()

    # Looping through batches
    torch_rmse = []
    for indx_batch, test_batch in enumerate(test_loader):

        results_dict = {} # initialize empty

        # calculating imputation error
        imputation_errors = calculate_imputation_error(var_info, test_batch, model, device, imputation_ratio)
        results_dict.update(imputation_errors)

        # calculating NLL and RMSE
        output, loss, nll = model.forward(test_batch, reconstruct=True, nll=True)
        # descaling
        if model.scale_type == 'outside_model':
            if model.scale == 'standardize':
                output = destand_num(model.var_info, output)
                test_batch = destand_num(model.var_info, test_batch)
            elif model.scale == 'normalize':
                output = denorm_num(model.var_info, output)
                test_batch = denorm_num(model.var_info, test_batch)

        results_dict['NLL'] = nll.item()
        rmse = calculate_RMSE(var_info, test_batch, output)
        for variable_type in rmse.keys():
            results_dict['RMSE_'+variable_type] = rmse[variable_type]

        # generating performance dataframe
        single_results_df = pd.DataFrame.from_dict([results_dict])
        results_df = pd.concat([results_df, single_results_df])

        # testing with pytorch
        torch_rmse.append(torch.sqrt(nn.MSELoss()(output, test_batch)))

    print('torch_rmse: ', sum(torch_rmse)/len(torch_rmse))
    return results_df.mean(axis=0)


def create_imputation_mask(batch, var_info, imputation_ratio=0.5):

    # Initializing imputation mask
    imputation_mask = np.ones(batch.shape)
    # Looping over observations
    for obs_idx, observation in enumerate(batch):

        # Random draw of variables to set zero - based on imputation ratio
        imputation_variables = np.random.choice(list(var_info.keys()), size=int(len(var_info) * imputation_ratio),
                                                replace=False)

        # Slicing indices - based on numerical / categorical
        imp_idx = 0
        # Looping through variables to find current slices
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

            # Only if variable is contained in the drawn imputed variables do we set to zero
            if var in imputation_variables:
                # Put into mask
                imputation_mask[obs_idx, :][idx1:idx2] = 0

    return torch.tensor(imputation_mask)

