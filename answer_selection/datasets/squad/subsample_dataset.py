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

def dump_indexes_relativize(idxsById,fn,sub_idxs):
  idxs = []
  output_file = open(fn,'w')
  for art_id, _idxs in idxsById.items():
    output_file.write('\n'.join([str(x) for x in _idxs]) + '\n')
    idxs.extend(_idxs)
  output_file.close()
  idxs.sort()
  
  new_sub_idxs = []
  for i,_id in enumerate(idxs):
    if _id in sub_idxs:
      new_sub_idxs.append(i)
  return new_sub_idxs



def main(raw_data):
  count = 0
  subsample_indexes = []
  idsByStory = defaultdict(list)
  for i,article in enumerate(raw_data):
    for paragraph in article["paragraphs"]:
      for qas in paragraph["qas"]:
        idsByStory[i].append(count)
        count += 1

  print("Spliting training set -> train' , val'")

  train_idxs = defaultdict(list)
  val_idxs = defaultdict(list)
  train_idxs_sub = set()
  val_idxs_sub = set()

  train_sampling_ratio = 0.1 # 10% -> 8k for val'
  train_sub_ratio = 0.10
  val_sub_ratio = 0.3
  for art_id,sample_idxs in idsByStory.items():
    nsamples = len(sample_idxs)
    to_take = int(train_sampling_ratio * nsamples)
    if len(sample_idxs) < 2:
      to_take = 1
    subsample_val = np.random.choice(sample_idxs,size=to_take,replace=False)
    subsample_train = [x for x in sample_idxs if x not in subsample_val]
    val_idxs[art_id] = subsample_val
    train_idxs[art_id] = subsample_train

    ## subsample now
    # train'
    to_take = int(len(train_idxs[art_id])*train_sub_ratio)
    to_take = max(to_take,1)
    subsample_train = np.random.choice(train_idxs[art_id],size=to_take,replace=False)
    train_idxs_sub.update(subsample_train)
    # val'
    to_take = int(len(val_idxs[art_id])*val_sub_ratio)
    to_take = max(to_take,1)
    subsample_val = np.random.choice(val_idxs[art_id],size=to_take,replace=False)
    val_idxs_sub.update(subsample_val)
  #END-FOR
  ## debug
  print("\tNew training set size:",sum([len(y) for x,y in train_idxs.items()]))
  print("\tNew validation set size:",sum([len(y) for x,y in val_idxs.items()]))
  
  print("\tSubsampled : train: (%d) | val(%d)" % (len(train_idxs_sub),len(val_idxs_sub)) )

  ## write idxs 
  new_train_fn = os.path.join(PREPROC_DATA_DIR,'squad',"training.indexes")
  new_val_fn   = os.path.join(PREPROC_DATA_DIR,'squad',"validation.indexes")
  
  train_idxs_sub = dump_indexes_relativize(train_idxs,new_train_fn,train_idxs_sub)
  val_idxs_sub = dump_indexes_relativize(val_idxs,new_val_fn,val_idxs_sub)

  ## dump subsampling indexes now
  new_train_sub_fn = os.path.join(PREPROC_DATA_DIR,'squad',"training.subsampling_indexes")
  open(new_train_sub_fn,'w').write('\n'.join([str(x) for x in train_idxs_sub]))

  new_val_sub_fn = os.path.join(PREPROC_DATA_DIR,'squad',"validation.subsampling_indexes")
  open(new_val_sub_fn,'w').write('\n'.join([str(x) for x in val_idxs_sub]))
  ###
  



if __name__ == "__main__":
  
  filename = os.path.join(DATA_JSON_DIR,"train-v1.1.json")
  raw_data = json.load(open(filename,'r'))
  raw_data = raw_data["data"]
  
  main(raw_data)
  
  """
  print("Validation set....")
  filename = os.path.join(DATA_JSON_DIR,"dev-v1.1.json")
  raw_data = json.load(open(filename,'r'))
  raw_data = raw_data["data"]
  validation_fn = os.path.join(PREPROC_DATA_DIR,'squad',"validation.subsampling_indexes")
  main_val(raw_data,validation_fn)
  """
