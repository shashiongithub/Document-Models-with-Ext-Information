####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

"""
generates cnt & wgt trec_format files from wiki_cnn [unfiltered]
"""

import os, sys
from wordvecs_authors import *
from utils import DATA_PICKLE_DIR,\
                  PREPROC_DATA_DIR,\
                  uploadObject

data,wvecs,max_sent_len = uploadObject(os.path.join(DATA_PICKLE_DIR,'wiki_cnn'),True)

pref = os.path.join(DATA_PICKLE_DIR,"wikiqa")
splits = ['training','validation','test']
count = 0
for idx,csp in enumerate(splits):
  if idx!=2:  continue
  cnt_out   = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".res_cnt"),'w')
  wgt_out   = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".res_wgt"),'w')
  for item in data:
    if item["split"] != idx+1:  continue
    qid = item["qid"]
    aid = item["aid"]
    cnt,wgt = item["features"]
    cnt_out.write("%d 0 %d 0 %.1f 0\n" % (qid,aid,cnt))
    wgt_out.write("%d 0 %d 0 %.6f 0\n" % (qid,aid,wgt))

    if count%1000==0:
      print "->",count
    count += 1
