"""modified from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer-web.py
by karljeon44
"""
import logging
import os
import shutil
import traceback
import warnings
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"

import gradio as gr
import numpy as np
import torch
from fairseq import checkpoint_utils

from lib.model.models import SynthesizerTrnMs768NSFsid
from lib.model.vc_infer_pipeline import VC
from lib.utils.config import Config
from lib.utils.misc_utils import load_audio, HUBERT_FPATH
from lib.utils.process_ckpt import merge

logging.getLogger("numba").setLevel(logging.WARNING)


now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()


hubert_model = None

def load_hubert():
  global hubert_model
  models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    # ["hubert_base.pt"],
    [HUBERT_FPATH],
    suffix="",
  )
  hubert_model = models[0]
  hubert_model = hubert_model.to(config.device)
  if config.is_half:
    hubert_model = hubert_model.half()
  else:
    hubert_model = hubert_model.float()
  hubert_model.eval()


weight_root = "weights"
index_root = "logs"
names = []
for name in os.listdir(weight_root):
  if name.endswith(".pth"):
    names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
  for name in files:
    if name.endswith(".index") and "trained" not in name:
      index_paths.append("%s/%s" % (root, name))


def vc_single(
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
  global tgt_sr, net_g, vc, hubert_model, version
  if input_audio_path is None:
    return "You need to upload an audio", None
  if isinstance(file_index, list):
    file_index = file_index[0] if len(file_index) > 0 else ""
  print(f"Inferring with sid `{sid}` from file index `{file_index}` with rate {index_rate}")
  f0_up_key = int(f0_up_key)
  try:
    audio = load_audio(input_audio_path, 16000)
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
      audio /= audio_max
    times = [0, 0, 0]
    if not hubert_model:
      load_hubert()
    if_f0 = cpt.get("f0", 1)
    audio_opt = vc.pipeline(
      hubert_model,
      net_g,
      sid,
      audio,
      input_audio_path,
      times,
      f0_up_key,
      f0_method,
      file_index,
      # file_big_npy,
      index_rate,
      if_f0,
      filter_radius,
      tgt_sr,
      resample_sr,
      rms_mix_rate,
      version,
      protect,
      f0_file=f0_file,
    )
    if tgt_sr != resample_sr >= 16000:
      tgt_sr = resample_sr
    if file_index and os.path.exists(file_index):
      index_info = "Using index:%s." % file_index
    else:
      index_info = "Index not used."
    return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (index_info, times[0], times[1], times[2],), (tgt_sr, audio_opt)
  except:
    info = info = traceback.format_exc()
    print(info)
    return info, (None, None)



# 一个选项卡全局只能有一个音色
def get_vc(sid, to_return_protect0, to_return_protect1):
  global n_spk, tgt_sr, net_g, vc, cpt, version
  if sid == "" or sid == []:
    global hubert_model
    if hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
      print("clean_empty_cache")
      del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
      hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

      ###楼下不这么折腾清理不干净
      if_f0 = cpt.get("f0", 1)
      version = cpt.get("version", "v2")
      if if_f0 == 1:
        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
      else:
        breakpoint()
        # net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
      del net_g, cpt

      if torch.cuda.is_available():
        torch.cuda.empty_cache()

      cpt = None
    return {"visible": False, "__type__": "update"}

  person = "%s/%s" % (weight_root, sid)
  print("loading %s" % person)
  cpt = torch.load(person, map_location="cpu")
  tgt_sr = cpt["config"][-2]
  cpt["config"][-4] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
  if_f0 = cpt.get("f0", 1)
  if if_f0 == 0:
    to_return_protect0 = to_return_protect1 = {
      "visible": False,
      "value": 0.5,
      "__type__": "update",
    }
  else:
    to_return_protect0 = {
      "visible": True,
      "value": to_return_protect0,
      "__type__": "update",
    }
    to_return_protect1 = {
      "visible": True,
      "value": to_return_protect1,
      "__type__": "update",
    }
  version = cpt.get("version", "v2")
  if if_f0 == 1:
    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
  else:
    breakpoint()
    # net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

  del net_g.enc_q

  print(net_g.load_state_dict(cpt["weight"], strict=False))
  net_g.eval().to(config.device)
  if config.is_half:
    print("Net G init as FP16")
    net_g = net_g.half()
  else:
    print("Net G init as FP32")
    net_g = net_g.float()

  vc = VC(tgt_sr, config)
  n_spk = cpt["config"][-4]
  return (
    {"visible": True, "maximum": n_spk, "__type__": "update"},
    to_return_protect0,
    to_return_protect1,
  )


def change_choices():
  names = []
  for name in os.listdir(weight_root):
    if name.endswith(".pth"):
      names.append(name)
  index_paths = []
  for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
      if name.endswith(".index") and "trained" not in name:
        index_paths.append("%s/%s" % (root, name))
  return {"choices": sorted(names), "__type__": "update"}, {
    "choices": sorted(index_paths),
    "__type__": "update",
  }


def clean():
  return {"value": "", "__type__": "update"}


with gr.Blocks() as app:
  gr.Markdown(
    value="This software is open source under the MIT license. The author does not have any control over the software. Users who use the software and distribute the sounds exported by the software are solely responsible. <br>If you do not agree with this clause, you cannot use or reference any codes and files within the software package. <b>LICENSE<b> for details."
  )
  with gr.Tabs():
    with gr.TabItem("Inference"):
      with gr.Row():
        sid0 = gr.Dropdown(label="Inference Voice", choices=sorted(names))
        refresh_button = gr.Button("Refresh voice list and index path", variant="primary")
        clean_button = gr.Button("Unload voice to save GPU Memory", variant="primary")
        spk_item = gr.Slider(
          minimum=0,
          maximum=109,
          step=1,
          label="Singer/Speaker ID",
          value=1,
          visible=False,
          interactive=True,
        )
        clean_button.click(fn=clean, inputs=[], outputs=[sid0])
      with gr.Group():
        gr.Markdown(value="about +/- 12 key for gender conversion")
        with gr.Row():
          with gr.Column():
            vc_transform0 = gr.Number(label="Pitch Translation in int Semi-tones", value=0)
            input_audio0 = gr.Textbox(
              label="Path to Input Audio",
              value="haejimal_cover_40k.wav",
            )
            f0method0 = gr.Radio(
              label="Pitch Extraction Algorithm",
              choices=["pm", "harvest", "crepe", "mangio", "rmvpe"],
              value="rmvpe",
              interactive=True,
            )
            filter_radius0 = gr.Slider(
              minimum=0,
              maximum=7,
              label="If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.",
              value=3,
              step=1,
              interactive=True,
            )
          with gr.Column():
            resample_sr0 = gr.Slider(
              minimum=0,
              maximum=48000,
              label="Target Sample Rate (0 if no resampling)",
              value=0,
              step=1,
              interactive=True,
            )
            file_index2 = gr.Dropdown(
              label="Auto-detected Index List",
              choices=sorted(index_paths),
              interactive=True,
            )
            refresh_button.click(fn=change_choices, inputs=[], outputs=[sid0, file_index2])

            index_rate1 = gr.Slider(
              minimum=0,
              maximum=1,
              label="Feature Ratio (controls accent strength; high value may result in artifacts)",
              value=0.33,
              interactive=True,
            )
          with gr.Column():
            rms_mix_rate0 = gr.Slider(
              minimum=0,
              maximum=1,
              label="Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume",
              value=0.,
              interactive=True,
            )
            protect0 = gr.Slider(
              minimum=0,
              maximum=0.5,
              label="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy",
              value=0.33,
              step=0.01,
              interactive=True,
            )
          f0_file = gr.File(label="Optional f0 curve file in .csv")
          but0 = gr.Button("Convert", variant="primary")
          with gr.Row():
            vc_output1 = gr.Textbox(label="Output Information")
            vc_output2 = gr.Audio(label="Result Audio Playback")
          but0.click(
            vc_single,
            [
              spk_item,
              input_audio0,
              vc_transform0,
              f0_file,
              f0method0,
              file_index2,
              index_rate1,
              filter_radius0,
              resample_sr0,
              rms_mix_rate0,
              protect0,
            ],
            [vc_output1, vc_output2],
          )
        sid0.change(
          fn=get_vc,
          inputs=[sid0, protect0, protect0],
          outputs=[spk_item, protect0, protect0],
        )
      with gr.Group():
        gr.Markdown("Timbre Fusion")
        with gr.Row():
          ckpt_a = gr.Textbox(label="Path to Model A", value="", interactive=True)
          ckpt_b = gr.Textbox(label="Path to Model B", value="", interactive=True)
          alpha_a = gr.Slider(
            minimum=0,
            maximum=1,
            label="Alpha for Model A",
            value=0.5,
            interactive=True,
          )
        with gr.Row():
          sr_ = gr.Radio(
            label="Target Sample Rate",
            choices=["32k", "40k", "48k"],
            value="48k",
            interactive=True,
          )
          if_f0_ = gr.Radio(
            label="Whether models have Pitch Guidance",
            choices=["Yes", "No"],
            value="Yes",
            interactive=True,
          )
          info__ = gr.Textbox(
            label="Model Information", value="", max_lines=8, interactive=True
          )
          name_to_save0 = gr.Textbox(
            label="Fused model Name (without extension)",
            value="",
            max_lines=1,
            interactive=True,
          )
          version_2 = gr.Radio(
            label="Version",
            choices=["v1", "v2"],
            value="v2",
            interactive=True,
          )
        with gr.Row():
          but6 = gr.Button("Fuse", variant="primary")
          info4 = gr.Textbox(label="Output Information", value="", max_lines=8)
        but6.click(
          merge,
          [
            ckpt_a,
            ckpt_b,
            alpha_a,
            sr_,
            if_f0_,
            info__,
            name_to_save0,
            version_2,
          ],
          info4,
        )

app.launch(
  server_name="0.0.0.0",
  inbrowser=not config.noautoopen,
  server_port=config.listen_port,
  quiet=True,
)
