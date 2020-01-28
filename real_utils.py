"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: utils.py
about: all utilities
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure





def validation(net, val_data_loader, device, category, save_tag=True):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            haze, image_name = val_data
            haze = haze.to(device)
            dehaze = net(haze)


        # --- Save image --- #
        if save_tag:
            save_image(dehaze, image_name, category)
    return 0 

def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './{}_results_real/{}'.format(category, image_name[ind][:-3] + '.png'))


