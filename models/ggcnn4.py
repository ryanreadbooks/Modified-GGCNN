"""
written by ryanreadbooks
date: 2021/11/3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


# UNet作为网络主干的GGCNN网络
class GGCNN4(nn.Module):
    def __init__(self, input_channels=1):
        super(GGCNN4, self).__init__()

        self.features = UNet(input_channels, output_channels=16)

        self.pos_output = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.cos_output = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.sin_output = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.width_output = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        # p_loss = F.mse_loss(pos_pred, y_pos)
        p_loss = F.binary_cross_entropy_with_logits(pos_pred, y_pos)  # cross entropy loss
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        # width_loss = F.mse_loss(width_pred, y_width)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred.sigmoid(),
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
