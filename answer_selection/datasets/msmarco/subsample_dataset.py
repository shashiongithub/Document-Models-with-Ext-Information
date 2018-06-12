####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

'''
Subsamples data to ~half the size and
generates .doc and .question files, each with format
    <story_id>
    wid wid wid...
    wid wid ...

    <story_id>
    wid ...
    ...
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import pdb
import nltk
import io
from utils import *
from collections import defaultdict

np.random.seed(42)


if __name__ == "__main__":
  
  #train_gen = read_data("training")
  
  count = 0
  filename = os.path.join(DATA_JSON_DIR,'train_v1.1.json')
  data = []
  for line in open(filename,'r'):
    data.append(json.loads(line))

  for ms_sample in data:
    label_sum = 0
    for passage in ms_sample["passages"]:
      if "is_selected" in passage:
        label_sum += passage["is_selected"]
    if label_sum!=0:
      count += 1
    if count % 10000 == 0:
      print("->",count)
  
  
  val_idxs = np.random.choice(range(count),size=int(0.1*count),replace=False)
  train_idxs = [x for x in range(count) if x not in val_idxs]

  train_sub_idxs = np.random.choice(range(len(train_idxs)),size=int(0.12*len(train_idxs)),replace=False)
  val_sub_idxs = np.random.choice(range(len(val_idxs)),size=int(0.25*len(val_idxs)),replace=False)

  fn = os.path.join(PREPROC_DATA_DIR,'msmarco',"training.indexes")
  open(fn,'w').write('\n'.join([str(idx) for idx in train_idxs]))

  fn = os.path.join(PREPROC_DATA_DIR,'msmarco',"validation.indexes")
  open(fn,'w').write('\n'.join([str(idx) for idx in val_idxs]))

  fn = os.path.join(PREPROC_DATA_DIR,'msmarco',"training.subsampling_indexes")
  open(fn,'w').write('\n'.join([str(idx) for idx in train_sub_idxs]))

  fn = os.path.join(PREPROC_DATA_DIR,'msmarco',"validation.subsampling_indexes")
  open(fn,'w').write('\n'.join([str(idx) for idx in val_sub_idxs]))

  print("Training set: ",len(train_idxs))
  print("Validation set: ",len(val_idxs))
  print("----------")
  print("Training subsampled: ",len(train_sub_idxs))
  print("Validation subsampled: ",len(val_sub_idxs))
