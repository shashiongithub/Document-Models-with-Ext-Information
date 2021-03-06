import sys
sys.path.append('../../common')

import os
import numpy as np
import pdb
import sys
from collections import defaultdict
from utils_extra import *
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # ./

def reformat_dataset(labels,sc_data):
    n = len(labels)
    X = []
    Y = []
    doc_lens = []

    for i in range(n):
        doc_len = len(labels[i])
        meta_x = np.vstack([
            sc_data[i][:doc_len]
            ])
        X.append(meta_x)
        Y.extend(labels[i])
        doc_lens.append(doc_len)
    X = np.hstack(X).T
    Y = np.array(Y,dtype=np.float32)

    return X,Y,doc_lens


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", '-d', default="wikiqa", type=str, help="Dataset name")
  parser.add_argument("--mode", '-m', default="first", type=str, help="tie-breaking strategy [first,last]")
  args = parser.parse_args()

  dataset = args.dataset

  PREPROC_DATA_DIR = os.path.join("..",'datasets','preprocessed_data',dataset)

  splits = ['training','validation','test']
    
  FLAGS.pretrained_wordembedding_orgdata = "../datasets/word_emb/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec"
  FLAGS.preprocessed_data_directory = "../datasets/preprocessed_data"

  for corpus_split in splits:
    fn = os.path.join(PREPROC_DATA_DIR,corpus_split+".isf.scores")

    label_data_list      = open("%s/%s.label" % (PREPROC_DATA_DIR,corpus_split),'r').read().strip().split("\n\n") # Use collective oracle
    isf_scores_data_list = open("%s/%s.isf.scores" % (PREPROC_DATA_DIR,corpus_split),'r').read().strip().split("\n\n") # 
    idf_scores_data_list = open("%s/%s.idf.scores" % (PREPROC_DATA_DIR,corpus_split),'r').read().strip().split("\n\n") # 
    locisf_scores_data_list = open("%s/%s.locisf.scores" % (PREPROC_DATA_DIR,corpus_split),'r').read().strip().split("\n\n") # 
    cnt_scores_data_list    = open("%s/%s.cnt.scores" % (PREPROC_DATA_DIR,corpus_split),'r').read().strip().split("\n\n") # 
    if dataset=="wikiqa":
      wgt_scores_data_list    = open("%s/%s.wgtcnt.scores" % (PREPROC_DATA_DIR,corpus_split),'r').read().strip().split("\n\n") # 
  
  
    ndocs = len(label_data_list)
    labels_data = []
    isf_data = []
    idf_data = []
    locisf_data = []
    cnt_data = []
    wgt_data = []
    
    for doc_idx in range(ndocs):
      label_lines = label_data_list[doc_idx].strip().split("\n")
      isf_lines = isf_scores_data_list[doc_idx].strip().split("\n")
      idf_lines = idf_scores_data_list[doc_idx].strip().split("\n")
      locisf_lines = locisf_scores_data_list[doc_idx].strip().split("\n")
      cnt_lines = cnt_scores_data_list[doc_idx].strip().split("\n")
      if dataset=="wikiqa":
        wgt_lines = wgt_scores_data_list[doc_idx].strip().split("\n")

      labels = np.array([int(x) for x in label_lines[1:]])
      labels = (labels>0).astype(float) # newsqa has 0,1,2
      isf_scores = np.array([float(x) for x in isf_lines[1:]])
      idf_scores = np.array([float(x) for x in idf_lines[1:]])
      locisf_scores = np.array([float(x) for x in locisf_lines[1:]])
      cnt_scores = np.array([float(int(x)) for x in cnt_lines[1:]])
      if dataset=="wikiqa":
        wgt_scores = np.array([float(x) for x in wgt_lines[1:]])
      
      labels_data.append(labels)
      isf_data.append(isf_scores)
      idf_data.append(idf_scores)
      locisf_data.append(locisf_scores)
      cnt_data.append(cnt_scores)
      if dataset=="wikiqa":
        wgt_data.append(wgt_scores)
    ##

    ###
    pred,y,dls = reformat_dataset(labels_data,cnt_data)
    acc = accuracy(y,pred,dls,args.mode)
    mrr = mrr_metric(y,pred,dls,corpus_split,args.mode)
    _map = map_score(y,pred,dls,corpus_split,args.mode)
    print("Cnt baseline    : ACC:%.4f | MAP:%.4f | MRR:%.4f" % (acc,_map,mrr) )
    # print("Cnt baseline,%.4f,%.4f,%.4f" % (acc,_map,mrr) )

    
    if dataset=="wikiqa":
      pred,y,dls = reformat_dataset(labels_data,wgt_data)
      acc = accuracy(y,pred,dls,args.mode)
      mrr = mrr_metric(y,pred,dls,corpus_split,args.mode)
      _map = map_score(y,pred,dls,corpus_split,args.mode)
      print("Wgt Cnt baseline: ACC:%.4f | MAP:%.4f | MRR:%.4f" % (acc,_map,mrr) )
      # print("Wgt Cnt baseline,%.4f,%.4f,%.4f" % (acc,_map,mrr) )
    
    else:
      pred,y,dls = reformat_dataset(labels_data,idf_data)
      acc = accuracy(y,pred,dls,args.mode)
      mrr = mrr_metric(y,pred,dls,corpus_split,args.mode)
      _map = map_score(y,pred,dls,corpus_split,args.mode)
      print("IDF baseline    : ACC:%.4f | MAP:%.4f | MRR:%.4f" % (acc,_map,mrr) )
      # print("IDF baseline,%.4f,%.4f,%.4f" % (acc,_map,mrr) )

    
    pred,y,dls = reformat_dataset(labels_data,locisf_data)
    acc = accuracy(y,pred,dls,args.mode)
    mrr = mrr_metric(y,pred,dls,corpus_split,args.mode)
    _map = map_score(y,pred,dls,corpus_split,args.mode)
    print("Loc-ISF baseline: ACC:%.4f | MAP:%.4f | MRR:%.4f" % (acc,_map,mrr) )
    # print("Loc-ISF baseline,%.4f,%.4f,%.4f" % (acc,_map,mrr) )
    
      
    pred,y,dls = reformat_dataset(labels_data,isf_data)
    acc = accuracy(y,pred,dls,args.mode)
    mrr = mrr_metric(y,pred,dls,corpus_split,args.mode)
    _map = map_score(y,pred,dls,corpus_split,args.mode)
    print("ISF baseline    : ACC:%.4f | MAP:%.4f | MRR:%.4f" % (acc,_map,mrr) )
    # print("ISF baseline,%.4f,%.4f,%.4f" % (acc,_map,mrr) )
    

    print
