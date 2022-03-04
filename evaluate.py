import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

def evaluate(net, dataloader, device, save_results=False, npz_name="", thresh=0.5):

    if save_results is True and npz_name is "":
        print("evaluate: Saving results but no npz name given. No results will be saved")
        save_results = False

    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # NOTE: this data saving assumes that we have only one image per batch
    image_pool = np.zeros((1,1,1))
    mask_pool = np.zeros((1,1,1))

    # iterate over the validation set
    for (i,batch) in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > thresh).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_dice = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_dice[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            if save_results:
                # make a copy of the image
                npimage = np.squeeze(image.cpu().detach().numpy())
                npmask = np.squeeze(torch.sigmoid(mask_pred).cpu().detach().numpy())
                #import matplotlib.pyplot as plt
                #plt.subplot(1,2,1)
                #plt.imshow(npmask[0,:])
                #plt.subplot(1,2,2)
                #plt.imshow(npmask[1,:])
                #plt.show()

                npmask = npmask[1,:]
                
                if image_pool.shape[1:] != npimage.shape:
                    image_pool = np.zeros(tuple([num_val_batches]) + npimage.shape)

                if mask_pool.shape[1:] != npmask.shape:
                    mask_pool = np.zeros(tuple([num_val_batches]) + npmask.shape)

                image_pool[i] = npimage
                mask_pool[i] = npmask

    if save_results:       
        print("source shape", image_pool.shape)
        print("prediction shape", mask_pool.shape)
        np.savez(npz_name, X=np.expand_dims(image_pool, -1), y=np.expand_dims(mask_pool, -1))

    net.train()
    return dice_score / num_val_batches
