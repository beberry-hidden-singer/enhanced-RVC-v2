# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.
"""from https://github.com/PlayVoice/NSF-BigVGAN/blob/main/model/alias/act.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sin, pow
from torch.nn import Parameter

from model.alias.resample import UpSample1d, DownSample1d


class SnakeBeta(nn.Module):
  '''
  A modified Snake function which uses separate parameters for the magnitude of the periodic components
  Shape:
      - Input: (B, C, T)
      - Output: (B, C, T), same shape as the input
  Parameters:
      - alpha - trainable parameter that controls frequency
      - beta - trainable parameter that controls magnitude
  References:
      - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
      https://arxiv.org/abs/2006.08195
  Examples:
      >>> a1 = snakebeta(256)
      >>> x = torch.randn(256)
      >>> x = a1(x)
  '''

  def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
    '''
    Initialization.
    INPUT:
        - in_features: shape of the input
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
        alpha is initialized to 1 by default, higher values = higher-frequency.
        beta is initialized to 1 by default, higher values = higher-magnitude.
        alpha will be trained along with the rest of your model.
    '''
    super(SnakeBeta, self).__init__()
    self.in_features = in_features
    # initialize alpha
    self.alpha_logscale = alpha_logscale
    if self.alpha_logscale:  # log scale alphas initialized to zeros
      self.alpha = Parameter(torch.zeros(in_features) * alpha)
      self.beta = Parameter(torch.zeros(in_features) * alpha)
    else:  # linear scale alphas initialized to ones
      self.alpha = Parameter(torch.ones(in_features) * alpha)
      self.beta = Parameter(torch.ones(in_features) * alpha)
    self.alpha.requires_grad = alpha_trainable
    self.beta.requires_grad = alpha_trainable
    self.no_div_by_zero = 0.000000001

  def forward(self, x):
    '''
    Forward pass of the function.
    Applies the function to the input elementwise.
    SnakeBeta = x + 1/b * sin^2 (xa)
    '''
    alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
    beta = self.beta.unsqueeze(0).unsqueeze(-1)
    if self.alpha_logscale:
      alpha = torch.exp(alpha)
      beta = torch.exp(beta)
    x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
    return x


class SnakeAlias(nn.Module):
  def __init__(self,
               channels,
               up_ratio: int = 2,
               down_ratio: int = 2,
               up_kernel_size: int = 12,
               down_kernel_size: int = 12,
               C = None):
    super().__init__()
    self.up_ratio = up_ratio
    self.down_ratio = down_ratio
    self.act = SnakeBeta(channels, alpha_logscale=True)
    self.upsample = UpSample1d(up_ratio, up_kernel_size,C=C)
    self.downsample = DownSample1d(down_ratio, down_kernel_size, C=C)

  # x: [B,C,T]
  def forward(self, x, C=None):
    x = self.upsample(x, C)
    x = self.act(x)
    x = self.downsample(x)
    return x


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
  shape = x.shape
  x = x.reshape(shape[0], shape[1], -1)
  x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
  x = x.reshape(shape)
  return x


class Snake1d(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.alpha = nn.Parameter(torch.ones(1, channels, 1))
    # torch.nn.init.xavier_uniform_(self.alpha)


  def forward(self, x):
    return snake(x, self.alpha)
