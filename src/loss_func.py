import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, input, target):
        """
        Combines Binary Cross Entropy (BCE) loss with Dice loss.

        Parameters:
        input: Predicted output with shape (B, C, H, W).
               The input must be passed through a sigmoid activation to
               constrain values between 0 and 1.
        target: Ground truth mask with shape (B, C, H, W).

        References:
        - https://arxiv.org/pdf/1807.10165.pdf
        - https://github.com/MrGiovanni/UNetPlusPlus/blob/dbe3806d7e859f1691c5f7816e756923fd0786b7/helper_functions.py#L47
        """
        bce_loss = self.bce(input, target.float())
        dice_loss = self.dice(input, target)

        loss = self.alpha * bce_loss + self.beta * dice_loss
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        """
        Computes the Dice Loss between the predicted output and the ground truth mask.

        Parameters:
        input: Predicted output with shape (B, C, H, W).
               The input must be passed through a sigmoid activation to
               constrain values between 0 and 1.
        target: Ground truth mask with shape (B, C, H, W).

        References:
        - https://github.com/MrGiovanni/UNetPlusPlus/blob/dbe3806d7e859f1691c5f7816e756923fd0786b7/helper_functions.py#L44
        """
        batch_size = target.size(0)
        probs_flat = input.view(batch_size, -1)
        target_flat = target.view(batch_size, -1).float()

        # dice coefficent
        intersection = (probs_flat * target_flat).sum(dim=1)
        union = (probs_flat**2).sum(dim=1) + (target_flat**2).sum(dim=1)
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff

        return dice_loss.mean()
