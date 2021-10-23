import torch
import torch.nn as nn
import torch.nn.functional as F

from .dla_models import *


class GGCNN3(nn.Module):
    def __init__(self, input_channels=1, backend='dla34up'):
        super().__init__()

        self.features = dlaup_func_dict[backend](16, input_channels=input_channels)
        
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

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }


