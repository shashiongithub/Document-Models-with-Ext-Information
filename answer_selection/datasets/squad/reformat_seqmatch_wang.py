####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import pdb
import io
from utils import *
from collections import defaultdict

sampling_ratio = 0.1
np.random.seed(42)


def load_subsample_idxs(csplit):
  fn = os.path.join(PREPROC_DATA_DIR,'squad',csplit+".subsampling_indexes")
  idxs = [int(x) for x in open(fn,'r').read().strip('\n').split('\n')]
  return idxs


if __name__ == "__main__":
    if len(sys.argv)>2:
        print("Usage: python reformat_seqmatch_wang.py [-subsample]")
        sys.exit(1)
    subsample = False
    if len(sys.argv)==2:
        subsample = True if sys.argv[1]=='-subsample' else False

    train_idxs = open(os.path.join(PREPROC_DATA_DIR,'squad',"training.indexes"),'r').read().strip('\n').split('\n')
    val_idxs   = open(os.path.join(PREPROC_DATA_DIR,'squad',"validation.indexes"),'r').read().strip('\n').split('\n')
    train_idxs = [int(x) for x in train_idxs]
    val_idxs = [int(x) for x in val_idxs]

    data_gen = read_data("training")
    data = []
    for sample in data_gen:
        data.append(sample)
    data_gen = read_data("validation")
    data_test = []
    for sample in data_gen:
        data_test.append(sample)

    splits = ['training','validation','test']

    for idxs, corpus_split in zip([train_idxs,val_idxs,range(len(data_test))],splits):
        output_file = ''
        sub_idxs = range(100000)
        pref = 'squad_as'
        if subsample and corpus_split!="test":
            pref = 'squad_as_sub'
            sub_idxs = load_subsample_idxs(corpus_split)
        output_file = io.open(os.path.join(DATASETS_BASEDIR,
                            "../SeqMatchSeq_wang/data",pref)+"/"+corpus_split+".txt",
                            mode='w',encoding='utf-8')

        print("---dumping ",corpus_split)
        idx = -1
        counter = 0
        
        for _id in idxs:
            if corpus_split=='test':
                sample_id,sents,question,labels = data_test[_id].unpack()
            else:
                sample_id,sents,question,labels = data[_id].unpack()

            idx += 1
            if idx not in sub_idxs:
                continue
            question = ' '.join(question)

            if labels.shape[0] != len(labels):
                pdb.set_trace()

            if labels.sum() == 0:
                print("empty: ",idx)
        
            counter += len(labels)
            for i in range(labels.shape[0]):
                output_file.write("%d\t%s\t%s\t%d\n" % (idx+1,question,' '.join(sents[i]),labels[i]) )
        
        output_file.close()
        print("lines: ",counter)

