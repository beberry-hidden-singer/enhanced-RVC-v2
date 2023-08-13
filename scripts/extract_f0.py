"""modified from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/extract_f0_print.py
by karljeon44
"""
import argparse
import logging
import os
import traceback
from multiprocessing import Process

import numpy as np
import parselmouth
import pyworld
import torch
import torchcrepe

from utils.misc_utils import get_device, load_audio, RMVPE_FPATH

logging.getLogger("numba").setLevel(logging.WARNING)


argparser = argparse.ArgumentParser()
argparser.add_argument('exp_dir', help='experiment dirpath')
argparser.add_argument('-n', '--num_proc', type=int, default=4, help='number of processes to use')
argparser.add_argument('-f', '--f0_method', type=str.lower, default='rmvpe', help='which f0 extraction algorithm to use',
                       choices=['pm', 'harvest', 'dio', 'crepe', 'mangio', 'mangio-crepe', 'rmvpe'])
argparser.add_argument('--crepe_batch_size', type=int, default=512, help='batch size when using CREPE')
argparser.add_argument('--mangio_hop_length', type=int, default=128, help='hop length when using MANGIO-CREPE')

args= argparser.parse_args()
exp_dir = args.exp_dir
n_p = args.num_proc
f0method = args.f0_method

crepe_batch_size_or_hop_length = 0
if f0method == 'crepe':
  crepe_batch_size_or_hop_length = args.crepe_batch_size
  print("CREPE Batch Size:", crepe_batch_size_or_hop_length)
elif 'mangio' in f0method:
  crepe_batch_size_or_hop_length = args.mangio_hop_length
  print("Mangio-CREPE Hop Length:", crepe_batch_size_or_hop_length)

device = get_device()

class FeatureInput(object):
  def __init__(self, samplerate=16000, hop_size=160, is_half=False):
    self.fs = samplerate
    self.hop = hop_size

    self.f0_bin = 256
    self.f0_max = 1100.0
    self.f0_min = 50.0
    self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
    self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    self.rmvpe = None
    self.is_half = is_half  # TODO: make this customizable



  def compute_f0(self, path, f0_method, batch_size_or_hop_length):
    x = load_audio(path, self.fs)
    p_len = x.shape[0] // self.hop
    if f0_method == "pm":
      time_step = 160 / 16000 * 1000
      f0_min = 50
      f0_max = 1100
      f0 = (
        parselmouth.Sound(x, self.fs)
        .to_pitch_ac(
          time_step=time_step / 1000,
          voicing_threshold=0.6,
          pitch_floor=f0_min,
          pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
      )
      pad_size = (p_len - len(f0) + 1) // 2
      if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
    elif f0_method == "harvest":
      f0, t = pyworld.harvest(
        x.astype(np.double),
        fs=self.fs,
        f0_ceil=self.f0_max,
        f0_floor=self.f0_min,
        frame_period=1000 * self.hop / self.fs,
      )
      f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
    elif f0_method == "dio":
      f0, t = pyworld.dio(
        x.astype(np.double),
        fs=self.fs,
        f0_ceil=self.f0_max,
        f0_floor=self.f0_min,
        frame_period=1000 * self.hop / self.fs,
      )
      f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

    elif f0_method == 'crepe':
      # Pick a batch size that doesn't cause memory errors on your gpu

      model = "full"
      # batch_size = 512
      # Compute pitch using first gpu
      audio = torch.tensor(np.copy(x))[None].float()
      f0, pd = torchcrepe.predict(
        audio,
        self.fs,
        160,
        self.f0_min,
        self.f0_max,
        model,
        batch_size=batch_size_or_hop_length,
        device=device,
        return_periodicity=True,
      )

      pd = torchcrepe.filter.median(pd, 3)
      f0 = torchcrepe.filter.mean(f0, 3)
      f0[pd < 0.1] = 0
      f0 = f0[0].cpu().numpy()

    elif f0_method in ['mangio', 'mangio-crepe']:
      # print("Performing crepe pitch extraction. (EXPERIMENTAL)")
      # print("CREPE PITCH EXTRACTION HOP LENGTH: " + str(crepe_hop_length))
      x = x.astype(np.float32)
      x /= np.quantile(np.abs(x), 0.999)

      audio = torch.from_numpy(x).to(device, copy=True)
      audio = torch.unsqueeze(audio, dim=0)
      if audio.ndim == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True).detach()
      audio = audio.detach()
      # print(
      #     "Initiating f0 Crepe Feature Extraction with an extraction_crepe_hop_length of: " +
      #     str(crepe_hop_length)
      # )
      # Pitch prediction for pitch extraction
      pitch = torchcrepe.predict(
        audio,
        self.fs,
        batch_size_or_hop_length,
        self.f0_min,
        self.f0_max,
        "full",
        batch_size=batch_size_or_hop_length * 2,
        device=device,
        pad=True
      )
      p_len = p_len or x.shape[0] // batch_size_or_hop_length
      # Resize the pitch
      source = np.array(pitch.squeeze(0).cpu().float().numpy())
      source[source < 0.001] = np.nan
      target = np.interp(
        np.arange(0, len(source) * p_len, len(source)) / p_len,
        np.arange(0, len(source)),
        source
      )
      f0 = np.nan_to_num(target)

    elif f0_method == 'rmvpe':
      if self.rmvpe is None:
        from model.rmvpe import RMVPE
        print("loading rmvpe model")
        self.rmvpe = RMVPE(RMVPE_FPATH, is_half=self.is_half, device=device)

      f0 = self.rmvpe.infer_from_audio(x, thred=0.03)

    else:
      raise ValueError(f'f0 method `{f0_method}` not understood')

    return f0

  def coarse_f0(self, f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1

    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(),f0_coarse.min())
    return f0_coarse

  def go(self, paths, f0_method, batch_size_or_hop_length):
    if len(paths) == 0:
      print("no-f0-todo")
    else:
      print("todo-f0-%s" % len(paths))
      n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
      for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
        try:
          if idx % n == 0:
            print("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
          if os.path.exists(opt_path1 + ".npy") == True and os.path.exists(opt_path2 + ".npy") == True:
            continue
          featur_pit = self.compute_f0(inp_path, f0_method, batch_size_or_hop_length)
          np.save(opt_path2, featur_pit, allow_pickle=False,)  # nsf
          coarse_pit = self.coarse_f0(featur_pit)
          np.save(opt_path1, coarse_pit, allow_pickle=False,)  # ori
        except:
          print("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))


if __name__ == "__main__":
  torch.multiprocessing.set_start_method('spawn')

  # exp_dir=r"E:\codes\py39\dataset\mi-test"
  # n_p=16
  # f = open("%s/log_extract_f0.log"%exp_dir, "w")
  featureInput = FeatureInput()
  paths = []
  inp_root = "%s/1_16k_wavs" % exp_dir
  opt_root1 = "%s/2a_f0" % exp_dir
  opt_root2 = "%s/2b-f0nsf" % exp_dir

  os.makedirs(opt_root1, exist_ok=True)
  os.makedirs(opt_root2, exist_ok=True)
  for name in sorted(list(os.listdir(inp_root))):
    inp_path = "%s/%s" % (inp_root, name)
    if "spec" in inp_path:
      continue
    opt_path1 = "%s/%s" % (opt_root1, name)
    opt_path2 = "%s/%s" % (opt_root2, name)
    paths.append([inp_path, opt_path1, opt_path2])

  ps = []
  for i in range(n_p):
    p = Process(target=featureInput.go, args=(paths[i::n_p], f0method, crepe_batch_size_or_hop_length))
    ps.append(p)
    p.start()
  for i in range(n_p):
    ps[i].join()
