import numpy as np
import matplotlib.pyplot as plt
import torch
from models import VAE


def get_model(model_name, total_num_vals, L, var_info, D, M, natural, device):

    if model_name == 'VAE':
        return VAE(total_num_vals=total_num_vals, L=L, var_info=var_info, D=D, M=M, natural=natural, device=device)
    elif model_name == 'BASELINE':
        return 'implement baseline here'
    else:
        raise NotImplementedError("Specified model is currently not implemented.")

def load_model(model_path, model):
    state_dict = torch.load(model_path + 'best.ckpt')
    model.load_state_dict(state_dict)
    model.eval()
    return model


#def evaluation(test_loader, var_info, model, model_best=None, epoch=None, M=256,natural=False,device=None):
def evaluation(model, data_loader, device):

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

    loss = 0.
    N = 0.
    for indx_batch, batch in enumerate(data_loader):
        batch = batch.to(device)
        #test_batch = torch.stack(test_batch[1]).float() # TODO: To access only one attribute - only needed as long as no multi-head
        # TODO: adjust to normal batch
        #numerical = test_batch[0].float()
        #categorical = test_batch[1]

        # concatenate into big input
        #test_batch = torch.cat((numerical, categorical), dim=1)


        # test_batch = test_batch[1]
        # TODO: this was also implemented in train.training and utils.samples_generated
        loss_t = model.forward(batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + batch.shape[0]
    loss = loss / N

    #if epoch is None:
    #    print(f'FINAL LOSS: nll={loss}')
    #else:
    #    print(f'Epoch: {epoch}, val nll={loss}')

    return loss

#def evaluate_to_table(test_loader, var_info, name=None, model_best=None, epoch=None, M=256,natural=False,device=None):
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
        loss_t = model.forward(batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + batch.shape[0]
    loss = loss / N

    '''
    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')
    '''

    return loss # + mse? (NLL, MSE)


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    #TODO: same deal as in samples_generated, evaluation and train.training
    x = torch.stack(next(iter(test_loader))[1]).float().detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(save_path, name, data_loader, extra_name=''):
    # TODO: originally:
    # x = next(iter(data_loader)).detach().numpy()
    #x = torch.stack(next(iter(data_loader))[1]).float().detach().numpy()
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


def plot_curve(name):#, nll_val):
    #plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


#def get_test_results(nll_val, result_path, test_loader,var_info, D=256, natural=None, device=None):
#    test_loss = evaluation(test_loader, var_info, name=result_path, D=D, natural=natural,device=device)
#    f = open(result_path + '_test_loss.txt', "w")

def get_test_results(model, result_path, model_name, test_loader, var_info, device):
    # loading best model
    model = load_model(model_path=result_path, model=model)
    test_loss = evaluation(model, test_loader, device)
    f = open(result_path + 'test_loss.txt', "w")
    f.write(str(test_loss))
    f.close()

    # samples_real(result_path, test_loader)

    plot_curve(result_path) #, nll_val)