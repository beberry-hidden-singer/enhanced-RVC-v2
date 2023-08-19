#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 7/17/23 6:12 AM
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from torch.nn.utils import spectral_norm, weight_norm

from model import modules
from model.commons import get_padding

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


### MSSTFT Discriminator from https://github.com/facebookresearch/encodec/blob/main/encodec/msstftd.py
class ConvLayerNorm(nn.LayerNorm):
  """
  Convolution-friendly LayerNorm that moves channels to last dimensions
  before running the normalization and moves them back to original position right after.
  """

  def __init__(self, normalized_shape, **kwargs):
    super().__init__(normalized_shape, **kwargs)

  def forward(self, x):
    x = rearrange(x, 'b ... t -> b t ...')
    x = super().forward(x)
    x = rearrange(x, 'b t ... -> b ... t')
    return x


class NormConv2d(nn.Module):
  """Wrapper around Conv2d and normalization applied to this conv
  to provide a uniform interface across normalization approaches.
  """

  def __init__(self, *args, norm: str = 'none', norm_kwargs = {}, **kwargs):
    super().__init__()

    self.conv = nn.Conv2d(*args, **kwargs)

    CONV_NORMALIZATIONS = ['none', 'weight_norm', 'spectral_norm', 'time_layer_norm', 'layer_norm', 'time_group_norm']
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        self.conv = weight_norm(self.conv)
    elif norm == 'spectral_norm':
        self.conv = spectral_norm(self.conv)

    # self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
    self.norm = nn.Identity()
    if norm == 'layer_norm':
        assert isinstance(self.conv, nn.modules.conv._ConvNd)
        self.norm = ConvLayerNorm(self.conv.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        assert isinstance(self.conv, nn.modules.conv._ConvNd)
        self.norm = nn.GroupNorm(1, self.conv.out_channels, **norm_kwargs)

    self.norm_type = norm

  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    return x


def get_2d_padding(kernel_size, dilation = (1, 1)):
  return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
  """STFT sub-discriminator.
  Args:
      filters (int): Number of filters in convolutions
      in_channels (int): Number of input channels. Default: 1
      out_channels (int): Number of output channels. Default: 1
      n_fft (int): Size of FFT for each scale. Default: 1024
      hop_length (int): Length of hop between STFT windows for each scale. Default: 256
      kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
      stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
      dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
      win_length (int): Window size for each scale. Default: 1024
      normalized (bool): Whether to normalize by magnitude after stft. Default: True
      norm (str): Normalization method. Default: `'weight_norm'`
      activation (str): Activation function. Default: `'LeakyReLU'`
      activation_params (dict): Parameters to provide to the activation function.
      growth (int): Growth factor for the filters. Default: 1
  """

  def __init__(self, filters, in_channels: int = 1, out_channels: int = 1, n_fft: int = 1024, hop_length: int = 256,
               win_length: int = 1024, max_filters: int = 1024, filters_scale: int = 1, kernel_size = (3, 9), dilations = [1, 2, 4],
               stride = (1, 2), normalized: bool = True, norm: str = 'weight_norm', activation: str = 'LeakyReLU',
               activation_params: dict = {'negative_slope': 0.2}):
    super().__init__()
    assert len(kernel_size) == 2
    assert len(stride) == 2
    self.filters = filters
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.normalized = normalized
    self.activation = getattr(torch.nn, activation)(**activation_params)
    self.spec_transform = torchaudio.transforms.Spectrogram(
      n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
      normalized=self.normalized, center=False, pad_mode=None, power=None)
    spec_channels = 2 * self.in_channels
    self.convs = nn.ModuleList()
    self.convs.append(
      NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
    )
    in_chs = min(filters_scale * self.filters, max_filters)
    for i, dilation in enumerate(dilations):
      out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
      self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=(dilation, 1),
                                   padding=get_2d_padding(kernel_size, (dilation, 1)), norm=norm))
      in_chs = out_chs
    out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
    self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                 padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm))
    self.conv_post = NormConv2d(out_chs, self.out_channels,
                                kernel_size=(kernel_size[0], kernel_size[0]),
                                padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm)

  def forward(self, x: torch.Tensor):
    fmap = []
    z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
    z = torch.cat([z.real, z.imag], dim=1)
    z = rearrange(z, 'b c w t -> b c t w')
    for i, layer in enumerate(self.convs):
      z = layer(z)
      z = self.activation(z)
      fmap.append(z)
    z = self.conv_post(z)
    return [(z, fmap)]


class MultiScaleSTFTDiscriminator(nn.Module):
  """Multi-Scale STFT (MS-STFT) discriminator.
  Args:
      filters (int): Number of filters in convolutions
      in_channels (int): Number of input channels. Default: 1
      out_channels (int): Number of output channels. Default: 1
      n_ffts (Sequence[int]): Size of FFT for each scale
      hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
      win_lengths (Sequence[int]): Window size for each scale
      **kwargs: additional args for STFTDiscriminator
  """

  def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
               # n_ffts = [1024, 2048, 512, 256, 128], hop_lengths = [256, 512, 128, 64, 32], win_lengths = [1024, 2048, 512, 256, 128],
               n_ffts = [2048, 4096, 2024, 512, 256], hop_lengths = [512, 1024, 256, 128, 64], win_lengths = [2048, 4096, 2024, 512, 256],
               **kwargs):
    super().__init__()
    assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
    self.discriminators = nn.ModuleList([
      DiscriminatorSTFT(
        filters, in_channels=in_channels, out_channels=out_channels, n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
      for i in range(len(n_ffts))
    ])
    self.num_discriminators = len(self.discriminators)

  def forward(self, x):
    ret = list()
    for disc in self.discriminators:
      disc_out = disc(x)
      if isinstance(disc_out, list):
        ret += disc_out
      else:
        ret.append(disc(x))

    return ret  # [(feat, score), (feat, score), (feat, score)]

  # def forward_(self, x: torch.Tensor):
  #   logits = []
  #   fmaps = []
  #   for disc in self.discriminators:
  #     logit, fmap = disc(x)
  #     logits.append(logit)
  #     fmaps.append(fmap)
  #   return logits, fmaps
