import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self,
                input,
                target, 
                alpha=0.5,
                beta=1.0,
                smooth=1e-5):
        '''
        Parameters:
        input: 輸入影像，shape 為 (B, H, W)
        target: label，shape 為 (B, H, W)

        References:
        https://arxiv.org/pdf/1807.10165.pdf
        https://github.com/MrGiovanni/UNetPlusPlus/blob/dbe3806d7e859f1691c5f7816e756923fd0786b7/helper_functions.py#L47/blob/dbe3806d7e859f1691c5f7816e756923fd0786b7/helper_functions.py#L47
        '''
        # binary cross entropy
        bce = F.binary_cross_entropy(input, target)

        # flatten
        input = input.view(-1)
        target = target.view(-1)
        # dice coefficent
        intersection = (input * target)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        return alpha * bce - beta * dice

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self,
                input,
                target, 
                smooth=1e-5,):
        '''
        Parameters:
        input: 輸入影像，shape 為 (B, H, W)
        target: label，shape 為 (B, H, W)

        References:
        https://github.com/MrGiovanni/UNetPlusPlus/blob/dbe3806d7e859f1691c5f7816e756923fd0786b7/helper_functions.py#L44
        '''
        # get batch
        batch_size = target.size(0)
        # get probability
        #input = torch.sigmoid(input)
        # flatten
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)

        # dice
        intersection = (input * target)
        dice = (2 * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)

        return (1 - dice).mean()