####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

import os,sys
import pdb
import io
import pandas as pd
from utils import *
import nltk

sampling_ratio = 0.5
np.random.seed(42)


def load_subsample_idxs(csplit):
  fn = os.path.join(PREPROC_DATA_DIR,'newsqa',csplit+".subsampling_indexes")
  idxs = [int(x) for x in open(fn,'r').read().strip('\n').split('\n')]
  return idxs


if __name__ == "__main__":
  if len(sys.argv)>2:
    print("Usage: python reformat_seqmatch_wang.py [-subsample]")
    sys.exit(1)
  subsample = False
  if len(sys.argv)==2:
    subsample = True if sys.argv[1]=='-subsample' else False
  splits = ['training','validation','test']

  for corpus_split in splits:
    data = read_data(corpus_split)
    sub_idxs = list(range(len(data.keys())))
    output_file = ''
    pref = 'newsqa'
    if subsample and corpus_split!="test":
      pref = 'newsqa_sub'
      sub_idxs = load_subsample_idxs(corpus_split)
    output_file = io.open(os.path.join(DATASETS_BASEDIR,"../SeqMatchSeq_wang/data",pref)+"/"+corpus_split+".txt",mode='w',encoding='utf-8')
    
    print("---dumping ",corpus_split)
    idx = 0
    for sample_id,sample in data.items():
      if idx not in sub_idxs:
        idx += 1
        continue
      _,sents,question,labels = sample.unpack()
      question = ' '.join(question)
      for i in range(labels.shape[0]):
        output_file.write("%d\t%s\t%s\t%d\n" % (idx+1,question,' '.join(sents[i]),labels[i]) )
      idx += 1
    output_file.close()

