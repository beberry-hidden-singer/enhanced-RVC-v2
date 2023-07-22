"""modified from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/extract_feature_print.py
by karljeon44
"""
import argparse
import os
import traceback

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# device=sys.argv[1]
argparser = argparse.ArgumentParser()
argparser.add_argument('exp_dir', help='experiment dirpath')
# argparser.add_argument('exp_dir', type=int, default=1)
# argparser.add_argument('exp_dir', type=int, default=0, help='experiment dirpath')

args= argparser.parse_args()
exp_dir = args.exp_dir
# n_part = int(sys.argv[2])
# i_part = int(sys.argv[3])

### ALAWYS CONSIDER ALL INPUT FILES

device = "cuda" if torch.cuda.is_available() else 'cpu'
f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
  print(strr)
  f.write("%s\n" % strr)
  f.flush()


model_path = "../pretrain/hubert_base.pt"

printt(exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = "%s/3_feature768" % exp_dir
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
  wav, sr = sf.read(wav_path)
  assert sr == 16000
  feats = torch.from_numpy(wav).float()
  if feats.dim() == 2:  # double channels
    feats = feats.mean(-1)
  assert feats.dim() == 1, feats.dim()
  if normalize:
    with torch.no_grad():
      feats = F.layer_norm(feats, feats.shape)
  feats = feats.view(1, -1)
  return feats


# HuBERT model
printt("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
  printt("Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main" % model_path)
  exit(0)
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="",)
model = models[0]
model = model.to(device)
printt("move model to %s" % device)
if device not in ["mps", "cpu"]:
  model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
  printt("no-feature-todo")
else:
  printt("all-feature-%s" % len(todo))
  for idx, file in enumerate(todo):
    try:
      if file.endswith(".wav"):
        wav_path = "%s/%s" % (wavPath, file)
        out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

        if os.path.exists(out_path):
          continue

        feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.half().to(device) if device not in ["mps", "cpu"] else feats.to(device),
          "padding_mask": padding_mask.to(device),
          "output_layer": 12,  # layer 9
        }
        with torch.no_grad():
          logits = model.extract_features(**inputs)
          feats = logits[0]

        feats = feats.squeeze(0).float().cpu().numpy()
        if np.isnan(feats).sum() == 0:
          np.save(out_path, feats, allow_pickle=False)
        else:
          printt("%s-contains nan" % file)
        if idx % n == 0:
          printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
    except:
      printt(traceback.format_exc())
  printt("all-feature-done")
