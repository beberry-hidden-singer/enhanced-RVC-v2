#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 7/17/23 6:12 AM
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from lib import modules
from lib.commons import get_padding

logger = logging.getLogger(__name__)


class Discriminator(nn.Module):
  """from https://github.com/PlayVoice/NSF-BigVGAN/blob/main/model/discriminator.py"""
  def __init__(self, resolutions, use_spectral_norm=False):
    super().__init__()

    self.MRD = MultiResolutionDiscriminator(resolutions, use_spectral_norm=use_spectral_norm)
    self.MPD = MultiPeriodDiscriminatorV2(use_spectral_norm=use_spectral_norm)

  def forward(self, x):
    r = self.MRD(x)
    p = self.MPD(x)
    # s = self.msd(x)  # now part of MPD
    return r + p


class DiscriminatorS(nn.Module):
  """from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/lib/infer_pack/models.py#L1036"""
  def __init__(self, use_spectral_norm=False):
    super(DiscriminatorS, self).__init__()
    norm_f = weight_norm if use_spectral_norm == False else spectral_norm
    self.convs = nn.ModuleList(
      [
        norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
        norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
        norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
        norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
        norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
        norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
      ]
    )
    self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

  def forward(self, x):
    fmap = []

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return [(x, fmap)]


class MultiPeriodDiscriminatorV2(torch.nn.Module):
  """from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/lib/infer_pack/models.py#L1006

  `MultiPeriodDiscriminatorV2` in original RVC (includes 23 and 37 as additional periods)
  """
  def __init__(self, use_spectral_norm=False):
    super(MultiPeriodDiscriminatorV2, self).__init__()
    # periods = [2, 3, 5, 7, 11, 17]  # V1
    periods = [2, 3, 5, 7, 11, 17, 23, 37]

    # self.discriminators = nn.ModuleList([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])
    self.discriminators = nn.ModuleList(
      [DiscriminatorS(use_spectral_norm=use_spectral_norm)] + \
      [DiscriminatorP(period, use_spectral_norm=use_spectral_norm) for period in periods]
    )

  def forward(self, x):
    ret = list()
    for disc in self.discriminators:
      disc_out = disc(x)
      if isinstance(disc_out, list):
        ret += disc_out
      else:
        ret.append(disc(x))

    return ret  # [(score, fmap), ...]

  def forward_org(self, y, y_hat):
    y_d_rs = []  #
    y_d_gs = []
    fmap_rs = []
    fmap_gs = []
    for i, d in enumerate(self.discriminators):
      disc_r = d(y)
      if isinstance(disc_r, list):
        disc_r = disc_r[0]
      y_d_r, fmap_r = disc_r

      disc_g = d(y_hat)
      if isinstance(disc_g, list):
        disc_g = disc_g[0]
      y_d_g, fmap_g = disc_g

      # for j in range(len(fmap_r)):
      #     print(i,j,y.shape,y_hat.shape,fmap_r[j].shape,fmap_g[j].shape)
      y_d_rs.append(y_d_r)
      y_d_gs.append(y_d_g)
      fmap_rs.append(fmap_r)
      fmap_gs.append(fmap_g)

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
  """From https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/lib/infer_pack/models.py#L1066"""
  def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
    super(DiscriminatorP, self).__init__()

    self.period = period
    self.use_spectral_norm = use_spectral_norm
    norm_f = weight_norm if use_spectral_norm == False else spectral_norm

    self.convs = nn.ModuleList([
      norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0), )),
      norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0), )),
      norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0), )),
      norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0), )),
      norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0), )),
    ])
    self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

  def forward(self, x):
    fmap = []

    # 1d to 2d
    b, c, t = x.shape
    if t % self.period != 0:  # pad first
      n_pad = self.period - (t % self.period)
      x = F.pad(x, (0, n_pad), "reflect")
      t = t + n_pad
    x = x.view(b, c, t // self.period, self.period)

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return x, fmap


class DiscriminatorR(torch.nn.Module):
  """from https://github.com/PlayVoice/NSF-BigVGAN/blob/main/model/mrd.py"""
  def __init__(self, resolution, use_spectral_norm=False):
    super(DiscriminatorR, self).__init__()

    self.resolution = resolution
    self.LRELU_SLOPE = modules.LRELU_SLOPE

    norm_f = weight_norm if use_spectral_norm == False else spectral_norm

    self.convs = nn.ModuleList([
      norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
      norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
      norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
      norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
      norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
    ])
    self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

  def forward(self, x):
    fmap = []

    x = self.spectrogram(x)
    x = x.unsqueeze(1)
    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, self.LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return x, fmap

  def spectrogram(self, x):
    n_fft, hop_length, win_length = self.resolution
    x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
    x = x.squeeze(1)
    x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=False)  # [B, F, TT, 2]
    mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

    return mag


class MultiResolutionDiscriminator(torch.nn.Module):
  """from https://github.com/PlayVoice/NSF-BigVGAN/blob/main/model/mrd.py"""
  def __init__(self, resolutions=None, use_spectral_norm=False):
    super(MultiResolutionDiscriminator, self).__init__()

    # (filter_length, hop_length, win_length)
    if not resolutions:
      resolutions = [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]
    self.resolutions = resolutions

    self.resolutions = resolutions
    self.discriminators = nn.ModuleList(
      [DiscriminatorR(resolution, use_spectral_norm=use_spectral_norm) for resolution in self.resolutions]
    )

  def forward(self, x):
    ret = list()
    for disc in self.discriminators:
      ret.append(disc(x))

    return ret  # [(feat, score), (feat, score), (feat, score)]
