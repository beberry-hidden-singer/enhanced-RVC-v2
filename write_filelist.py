#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 7/2/23 12:03 AM
import argparse
import logging
import os
from random import shuffle

logger = logging.getLogger(__name__)


def main(exp_dir, sample_rate, infer_spk_id=False):
  print("EXP Dir:", exp_dir)

  gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
  feature_dir = "%s/3_feature768" % exp_dir
  f0_dir = "%s/2a_f0" % (exp_dir)
  f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
  names = (
          set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
          & set([name.split(".")[0] for name in os.listdir(feature_dir)])
          & set([name.split(".")[0] for name in os.listdir(f0_dir)])
          & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
  )

  print("Num Names:", len(names))
  # print(names)


  speaker_mapping = dict()
  # speaker_mapping_count =

  opt = []
  unique_speaker_ids = []
  for name in names:
    speaker_id = name.split('_')[-1] if infer_spk_id else '0'
    # if spk_id_src not in speaker_mapping:
    #   speaker_mapping[spk_id_src] = len(speaker_mapping)
    # spk_id = speaker_mapping[spk_id_src]
    opt.append(
      "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
      % (
        gt_wavs_dir.replace("\\", "\\\\"),
        name,
        feature_dir.replace("\\", "\\\\"),
        name,
        f0_dir.replace("\\", "\\\\"),
        name,
        f0nsf_dir.replace("\\", "\\\\"),
        name,
        speaker_id,
      )
    )
    if speaker_id not in unique_speaker_ids:
      unique_speaker_ids.append(speaker_id)


  # print("Speaker Mapping:", speaker_mapping)
  # print(opt)
  # print(unique_speaker_ids)
  # with open('./speaker_mapping.json', 'w') as f:
  #   json.dump(speaker_mapping, f)
  print(len(opt))

  for spk_id in unique_speaker_ids:
    for _ in range(2):
      opt.append(
        "./logs/mute/0_gt_wavs/mute%s.wav|./logs/mute/3_feature%s/mute.npy|./logs/mute/2a_f0/mute.wav.npy|./logs/mute/2b-f0nsf/mute.wav.npy|%s" \
        % (sample_rate, 768, spk_id)
      )

  shuffle(opt)
  print(len(opt))
  with open("%s/filelist.txt" % exp_dir, "w") as f:
    f.write("\n".join(opt))
  print("write filelist done")


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('log_dir', help='dirpath to current log')
  argparser.add_argument('-sr', '--sample_rate', default='40k', help='target sample rate')
  argparser.add_argument('--infer_spk_id', action='store_true', help='whether to infer speaker ids from filenames')
  args = argparser.parse_args()

  main(args.log_dir, args.sample_rate, infer_spk_id=args.infer_spk_id)
