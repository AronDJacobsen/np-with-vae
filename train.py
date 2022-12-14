import torch
import numpy as np
from utils import evaluation, samples_generated

def training(logger, save_path, max_patience, num_epochs, model, optimizer, train_loader, val_loader, var_info, device):
    nll_val = []
    best_nll = 1000.
    patience = 0

    #save_path = "results/"

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(train_loader):
            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + torch.rand(batch.shape)
            batch = batch.to(device)
            _, loss, _ = model.forward(batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        #loss_val = evaluation(val_loader, var_info, model=model, model_best=model, epoch=e,natural=natural,device=device)
        loss_val = evaluation(model=model, data_loader=val_loader, device=device)
        logger.write_to_board(name="Validation", scalars={"NLL": loss_val}, index=e)
        print(f'Epoch: {e}, loss val={loss_val}')
        nll_val.append(loss_val.detach())  # save for plotting

        if e == 0:
            print('saved!')
            logger.log('saved!')
            torch.save(model.state_dict(), save_path + 'best.ckpt')

            #torch.save(model.state_dict(), f"{save_path}/best.ckpt")

            best_nll = loss_val
        else:
            if loss_val < best_nll: # saving the best models
                print('saved!')
                logger.log('saved!')
                torch.save(model.state_dict(), save_path + 'best.ckpt')
                best_nll = loss_val
                patience = 0

                # samples_generated(save_path + 'generated/', name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val