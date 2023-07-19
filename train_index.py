#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 7/18/23 12:24 PM
import argparse
import logging
import os
from multiprocessing import cpu_count

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


def train_index(log_dir):
  os.makedirs(log_dir, exist_ok=True)
  feature_dir = f"{log_dir}/3_feature768"
  assert os.path.exists(feature_dir)

  listdir_res = list(os.listdir(feature_dir))
  if len(listdir_res) == 0:
    raise ValueError()

  npys = []
  for name in sorted(listdir_res):
    phone = np.load("%s/%s" % (feature_dir, name))
    npys.append(phone)

  big_npy = np.concatenate(npys, 0)
  big_npy_idx = np.arange(big_npy.shape[0])
  np.random.shuffle(big_npy_idx)
  big_npy = big_npy[big_npy_idx]
  if big_npy.shape[0] > 2e5:
    print("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
    big_npy = (
      MiniBatchKMeans(
        n_clusters=10000,
        verbose=True,
        batch_size=256 * cpu_count(),
        compute_labels=False,
        init="random",
      )
      .fit(big_npy)
      .cluster_centers_
    )


  np.save("%s/total_fea.npy" % log_dir, big_npy)
  n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
  print("%s,%s" % (big_npy.shape, n_ivf))

  index = faiss.index_factory(768, "IVF%s,Flat" % n_ivf)
  # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
  print("training")
  index_ivf = faiss.extract_index_ivf(index)  #
  index_ivf.nprobe = 1
  index.train(big_npy)
  faiss.write_index(index, "%s/trained_IVF%s_Flat_nprobe_%s.index" % (log_dir, n_ivf, index_ivf.nprobe))

  # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
  print("adding")
  batch_size_add = 8192
  for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i: i + batch_size_add])
  faiss.write_index(index, "%s/added_IVF%s_Flat_nprobe_%s.index" % (log_dir, n_ivf, index_ivf.nprobe))
  print("Done.")


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('log_dir', help='dirpath to current log')
  args = argparser.parse_args()

  train_index(args.log_dir)
