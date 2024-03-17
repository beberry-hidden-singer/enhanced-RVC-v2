"""modified from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/train/utils.py
by karljeon44
"""
import argparse
import glob
import json
import logging
import os
import sys

import ffmpeg
import numpy as np
import torch
from scipy.io.wavfile import read

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

### constants
MATPLOTLIB_FLAG = False
PRETRAIN_DIR = './pretrain'
if not os.path.exists(PRETRAIN_DIR):
  PRETRAIN_DIR = '../pretrain'
assert os.path.exists(PRETRAIN_DIR)
HUBERT_FPATH = f'{PRETRAIN_DIR}/hubert/hubert_base.pt'
RMVPE_FPATH = f'{PRETRAIN_DIR}/rmvpe/model.pt'


def get_device():
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda'
  elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'
  print("Found Device:", device)
  return device


def load_audio(file, sr):
  try:
    # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
    # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
    # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
    file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    out, _ = (
      ffmpeg.input(file, threads=0)
      .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
      .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
  except Exception as e:
    raise RuntimeError(f"Failed to load audio: {e}")

  return np.frombuffer(out, np.float32).flatten()


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

  saved_state_dict = checkpoint_dict["model"]
  if hasattr(model, "module"):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict = {}
  for k, v in state_dict.items():  # 模型需要的shape
    try:
      new_state_dict[k] = saved_state_dict[k]
      if saved_state_dict[k].shape != state_dict[k].shape:
        print("shape-%s-mismatch|need-%s|get-%s" % (k, state_dict[k].shape, saved_state_dict[k].shape))  #
        raise KeyError
    except:
      # logger.info(traceback.format_exc())
      logger.info("%s is not in the checkpoint" % k)  # pretrain缺失的
      new_state_dict[k] = v  # 模型自带的随机值
  if hasattr(model, "module"):
    model.module.load_state_dict(new_state_dict, strict=False)
  else:
    model.load_state_dict(new_state_dict, strict=False)
  logger.info("Loaded model weights")

  iteration = checkpoint_dict["iteration"]
  learning_rate = checkpoint_dict["learning_rate"]
  if optimizer is not None and load_opt == 1:
    optimizer.load_state_dict(checkpoint_dict["optimizer"])

  logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at epoch {} to {}".format(iteration, checkpoint_path))
  if hasattr(model, "module"):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save(
    {
      "model": state_dict,
      "iteration": iteration,
      "optimizer": optimizer.state_dict(),
      "learning_rate": learning_rate,
    },
    checkpoint_path,
  )


def summarize(
        writer,
        global_step,
        scalars={},
        histograms={},
        images={},
        audios={},
        audio_sampling_rate=22050,
):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats="HWC")
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib

    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10, 2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib

    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(
    alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
  )
  fig.colorbar(im, ax=ax)
  xlabel = "Decoder timestep"
  if info is not None:
    xlabel += "\n\n" + info
  plt.xlabel(xlabel)
  plt.ylabel("Encoder timestep")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding="utf-8") as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams():
  parser = argparse.ArgumentParser()

  # core
  parser.add_argument("-e", "--experiment_dir", type=str, required=True, help="experiment dir")
  parser.add_argument("-sr", "--sample_rate", default='40k', type=str.lower, help="sample rate, 32k/40k/48k")
  parser.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size")
  parser.add_argument("-te", "--total_epoch", type=int, default=200, help="total_epoch")
  parser.add_argument('-se', "--save_every", type=int, default=5, help="checkpoint save frequency (epoch)")

  # pretrained
  parser.add_argument("-pg", "--pretrainG", type=str, default="", help="Pretrained Discriminator path")
  parser.add_argument("-pd", "--pretrainD", type=str, default="", help="Pretrained Generator path")
  parser.add_argument("-ps", "--pretrainS", type=str, default="", help="Pretrained Sovits-5.0 path")
  parser.add_argument("-pv", "--pretrainV", type=str, default="", help="Pretrained NSF-BigVGAN path")

  # flags
  parser.add_argument("--latest", action='store_true', help="whether to save the latest G/D pth file",)
  parser.add_argument("--cache", action='store_true', help="whether to cache the dataset in GPU memory",)
  parser.add_argument("--save_small_weights", action='store_true', help="save the extracted model in weights directory when saving checkpoints",)
  parser.add_argument("--no_f0", action='store_true', help="whether to not use f0")
  parser.add_argument("--pretrain", action='store_true', help="whether to turn on pre-training mode")
  parser.add_argument('--load_opt', action='store_true', help='whether to load optimizer when loading checkpoints')

  # custom
  parser.add_argument('--seed', type=int, default=1234, help='training seed')
  parser.add_argument('--c_mel', type=float, default=45., help='multiplier for mel loss')
  parser.add_argument('--c_kl', type=float, default=1.0, help='multiplier for KL Div loss')
  parser.add_argument('--c_stft', type=float, default=0.5, help='multiplier for MR-STFT loss')
  # parser.add_argument('--snake', action='store_true', help='whether to use Generator with Snake Activations')
  parser.add_argument('--mrd', action='store_true', help='whether to use Multi-Resolution Discriminator')
  parser.add_argument('--msstftd', action='store_true', help='whether to add Multi-Scale STFT Discriminator')
  parser.add_argument('--mrstft', action='store_true', help='whether to add Multi-Resolution STFT Loss term')
  parser.add_argument('--weighted_mrstft', action='store_true', help='whether to use weighted version of Multi-Resolution STFT Loss')


  args = parser.parse_args()

  experiment_dir = args.experiment_dir
  os.makedirs(experiment_dir, exist_ok=True)
  name = os.path.basename(experiment_dir)
  print("EXP Name:", name)

  with open(f"configs/{args.sample_rate}.json", "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  hparams.model_dir = hparams.experiment_dir = experiment_dir
  hparams.save_every_epoch = args.save_every
  hparams.name = name
  hparams.total_epoch = args.total_epoch
  hparams.pretrainG = args.pretrainG
  hparams.pretrainD = args.pretrainD

  ### custom pre-trained paths
  hparams.pretrainS = args.pretrainS
  hparams.pretrainV = args.pretrainV

  # default resolutions
  # hparams.resolutions = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
  hparams.resolutions = [(1024, 120, 600), (2048, 240, 1200), (4096, 480, 2400), (512, 50, 240)]

  # overwrite config with custom params
  hparams.train.seed = args.seed
  hparams.train.c_mel = args.c_mel
  hparams.train.c_kl = args.c_kl
  hparams.train.c_stft = args.c_stft
  # hparams.model.snake = args.snake
  hparams.model.msstftd = args.msstftd
  hparams.model.mrd = args.mrd
  hparams.model.mrstft = args.mrstft
  hparams.model.weighted_mrstft = args.weighted_mrstft

  hparams.version = 'v2'
  hparams.gpus = '0'  # control this explicitly through CUDA_VISIBLE_DEVICES env
  hparams.train.batch_size = args.batch_size
  hparams.sample_rate = args.sample_rate
  hparams.if_f0 = not args.no_f0
  hparams.if_latest = args.latest
  hparams.if_pretrain = args.pretrain
  hparams.load_opt = args.load_opt
  hparams.save_every_weights = args.save_small_weights
  hparams.if_cache_data_in_gpu = args.cache
  hparams.data.training_files = "%s/filelist.txt" % experiment_dir

  if get_device() == 'mps' and hparams.train.fp16_run:
    print("Turning off mixed precision for MPS Training")
    hparams.train.fp16_run = False

  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  return hparams


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


class HParams:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
