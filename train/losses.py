"""modified from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/train/losses.py
with additional modifications from https://github.com/Nitin4525/SpeechEnhancement/blob/master/loss.py for weighted MR-STFT
by karljeon44"""
import torch
from torch.nn import functional as F


def feature_loss(fmap_r, fmap_g, normalize=False):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))
  if normalize:
    loss /= len(fmap_r)
  return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1 - dr) ** 2)
    g_loss = torch.mean(dg**2)
    loss += r_loss + g_loss
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())
  return loss, r_losses, g_losses


def generator_loss(disc_outputs, normalize=False):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1 - dg) ** 2)
    gen_losses.append(l)
    loss += l
  if normalize:
    loss = loss / len(disc_outputs)
  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l


def stft(x, fft_size, hop_size, win_length, window):
  """Perform STFT and convert to magnitude spectrogram.
  Args:
      x (Tensor): Input signal tensor (B, T).
      fft_size (int): FFT size.
      hop_size (int): Hop size.
      win_length (int): Window length.
      window (str): Window function type.
  Returns:
      Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
  """
  x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
  real = x_stft[..., 0]
  imag = x_stft[..., 1]

  # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
  return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
  """Spectral convergence loss module."""

  def __init__(self):
    """Initilize spectral convergence loss module."""
    super(SpectralConvergengeLoss, self).__init__()

  def forward(self, x_mag, y_mag):
    """Calculate forward propagation.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Spectral convergence loss value.
    """
    return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
  """Log STFT magnitude loss module."""

  def __init__(self):
    """Initilize los STFT magnitude loss module."""
    super(LogSTFTMagnitudeLoss, self).__init__()

  def forward(self, x_mag, y_mag):
    """Calculate forward propagation.
    Args:
        x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
    Returns:
        Tensor: Log STFT magnitude loss value.
    """
    return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
  """STFT loss module."""

  def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", weight_by_factor=False):
    """Initialize STFT loss module."""
    super(STFTLoss, self).__init__()
    self.fft_size = fft_size
    self.shift_size = shift_size
    self.win_length = win_length
    self.window = getattr(torch, window)(win_length).cuda()
    self.spectral_convergenge_loss = SpectralConvergengeLoss()
    self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
    self.factor = fft_size / 2048 if weight_by_factor else 1.

  def forward(self, x, y):
    """Calculate forward propagation.
    Args:
        x (Tensor): Predicted signal (B, T).
        y (Tensor): Groundtruth signal (B, T).
    Returns:
        Tensor: Spectral convergence loss value.
        Tensor: Log STFT magnitude loss value.
    """
    x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
    y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
    sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
    mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

    # return sc_loss, mag_loss
    return sc_loss * self.factor, mag_loss * self.factor


class MultiResolutionSTFTLoss(torch.nn.Module):
  """Multi resolution STFT loss module."""

  def __init__(self, resolutions, window="hann_window", weight_by_factor=False):
    """Initialize Multi resolution STFT loss module.
    Args:
        resolutions (list): List of (FFT size, hop size, window length).
        window (str): Window function type.
    """
    super(MultiResolutionSTFTLoss, self).__init__()

    self.stft_losses = torch.nn.ModuleList()
    for fs, ss, wl in resolutions:
      self.stft_losses += [STFTLoss(fs, ss, wl, window, weight_by_factor=weight_by_factor)]

  def forward(self, x, y):
    """Calculate forward propagation.
    Args:
        x (Tensor): Predicted signal (B, T).
        y (Tensor): Groundtruth signal (B, T).
    Returns:
        Tensor: Multi resolution spectral convergence loss value.
        Tensor: Multi resolution log STFT magnitude loss value.
    """
    sc_loss = 0.0
    mag_loss = 0.0
    for f in self.stft_losses:
      sc_l, mag_l = f(x, y)
      sc_loss += sc_l
      mag_loss += mag_l

    sc_loss /= len(self.stft_losses)
    mag_loss /= len(self.stft_losses)

    return sc_loss, mag_loss
    # return sc_loss * self.factor_sc, mag_loss * self.factor_mag
