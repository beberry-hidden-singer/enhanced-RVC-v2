#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 7/23/23 5:05 AM
import argparse
import logging
import os

from utils.process_ckpt import extract_small_model

logger = logging.getLogger(__name__)


def main():
  """[!!!]currently does not process snake config when extracting"""
  argparser = argparse.ArgumentParser()
  argparser.add_argument('model', help='path to model A')
  argparser.add_argument('-n', '--name', required=True, help='name of the extracted small model')
  argparser.add_argument('-sr', '--sample_rate', default='48k', help='model sample rate')
  argparser.add_argument('--no_f0', action='store_true', help='whether the model does not use pitch guidance')
  # argparser.add_argument('--snake', action='store_true', help='whether to use Generator with Snake Activations')

  args = argparser.parse_args()
  assert os.path.exists(args.model)

  print(f"Extracting small model from `{args.model}`")
  print(extract_small_model(
    path=args.model,
    name=args.name,
    sr=args.sample_rate,
    if_f0=not args.no_f0,
    info='Extracted Model',
  ))


if __name__ == '__main__':
  main()
