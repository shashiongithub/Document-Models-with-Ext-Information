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

sampling_ratio = 0.5
np.random.seed(42)

def main(story_ids_csv,data_dict,output_fn):
  sid2Id = {}
  print("\tReading story ids...")
  for i,sid in enumerate(story_ids_csv["story_id"]):
    sid2Id[sid] = i
  ndata = len(data_dict["story_id"])
  idsByStory = {}
  hash_dict = []
  idx = 0
  print("\tCollecting indexes")
  for i in range(ndata):
    sid = data_dict["story_id"][i]
    question = data_dict["question"][i]
    sample_hash = get_hash(sid,question)

    if sample_hash not in hash_dict:
      hash_dict.append(sample_hash)
    else:
      print("\tRepeated!!:: ", sid,":: ",question)
      continue
    _id = sid2Id[sid]
    if _id not in idsByStory:
      idsByStory[_id] = []
    idsByStory[_id].append(idx)

    if idx%10000 == 0:
      print("\t->",idx)
    idx += 1

  print("\tSubsampling...")
  subsample_indexes = []
  for sid,sample_idxs in idsByStory.items():
    nsamples = len(sample_idxs)
    to_take = int(sampling_ratio * nsamples)
    if len(sample_idxs) <= 2:
      to_take = 1
    #to_take = 1 # take only one question  -> 12k
    subsample = np.random.choice(sample_idxs,size=to_take,replace=False)
    subsample_indexes.extend(subsample)

  output_file = open(output_fn,'w')
  output_file.write('\n'.join([str(idx) for idx in subsample_indexes]))
  print("\tComplete dataset size:",ndata)
  print("\tSubsampled dataset size:",len(subsample_indexes))



if __name__ == "__main__":
  """
  print("Training set....")
  story_ids_csv = pd.read_csv(os.path.join(MALUUBA_DIR,'train_story_ids.csv'),encoding='utf-8')
  data_dict = pd.read_csv(os.path.join(DATA_CSV_DIR,'train.csv'),encoding='utf-8')
  training_fn = os.path.join(PREPROC_DATA_DIR,'newsqa',"training.subsampling_indexes")
  main(story_ids_csv,data_dict,training_fn)
  """
  
  print("Validation set....")
  story_ids_csv = pd.read_csv(os.path.join(MALUUBA_DIR,'dev_story_ids.csv'),encoding='utf-8')
  data_dict = pd.read_csv(os.path.join(DATA_CSV_DIR,'dev.csv'),encoding='utf-8')
  validation_fn = os.path.join(PREPROC_DATA_DIR,'newsqa',"validation.subsampling_indexes")
  main(story_ids_csv,data_dict,validation_fn)

