####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

import os, sys
from utils import PREPROC_DATA_DIR, trail_id

pref = os.path.join(PREPROC_DATA_DIR,"newsqa")
for csp in ["training","validation","test"]:
  filtered_ref = ''
    
  label_lines = open(os.path.join(pref,csp+".org_ent.label"),'r').read().strip("\n").split("\n\n")
  cnts_lines = open(os.path.join(pref,csp+".cnt.scores"),'r').read().strip("\n").split("\n\n")
  #wgt_lines = open(os.path.join(pref,csp+".wgtcnt.scores"),'r').read().strip("\n").split("\n\n")

  idf_lines = open(os.path.join(pref,csp+".idf.scores"),'r').read().strip("\n").split("\n\n")
  isf_lines = open(os.path.join(pref,csp+".isf.scores"),'r').read().strip("\n").split("\n\n")
  locisf_lines = open(os.path.join(pref,csp+".locisf.scores"),'r').read().strip("\n").split("\n\n")

  label_out = open(os.path.join(pref,csp+".rel_info"),'w')
  cnt_out   = open(os.path.join(pref,csp+".res_cnt"),'w')
  #wgt_out   = open(os.path.join(pref,csp+".res_wgt"),'w')
  isf_out   = open(os.path.join(pref,csp+".res_isf"),'w')
  idf_out   = open(os.path.join(pref,csp+".res_idf"),'w')
  locisf_out   = open(os.path.join(pref,csp+".res_locisf"),'w')
  
  for qid,labels in enumerate(label_lines):
    labels = labels.split("\n")
    labels = [int(float(x)) for x in labels[1:]]
    cnts = [float(x) for x in cnts_lines[qid].split("\n")[1:]]
    #wgts = [float(x) for x in wgt_lines[qid].split("\n")[1:]]
    isfs = [float(x) for x in isf_lines[qid].split("\n")[1:]]
    idfs = [float(x) for x in idf_lines[qid].split("\n")[1:]]
    locisfs = [float(x) for x in locisf_lines[qid].split("\n")[1:]]
    for i,lab in enumerate(labels):
      label_out.write("%s 0 %s %d\n" % ( trail_id(qid+1),trail_id(i),lab) )
      cnt_out.write("%s\t0\t%s\t0\t%.1f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),cnts[i]))
      # wgt_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),wgts[i]))
      isf_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),isfs[i]))
      idf_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),idfs[i]))
      locisf_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),locisfs[i]))
    ##
  ##
  label_out.close()
  cnt_out.close()
  #wgt_out.close()
  isf_out.close()
  idf_out.close()
  locisf_out.close()

  ########################################
  ## subsampled set
  if csp=="test": continue
  indexes = open(os.path.join(pref,csp+".subsampling_indexes"),'r').read().strip("\n").split("\n")
  indexes = [int(x) for x in indexes]

  label_out = open(os.path.join(pref,csp+".rel_info_subsampled"),'w')
  cnt_out   = open(os.path.join(pref,csp+".res_cnt_subsampled"),'w')
  isf_out   = open(os.path.join(pref,csp+".res_isf_subsampled"),'w')
  idf_out   = open(os.path.join(pref,csp+".res_idf_subsampled"),'w')
  locisf_out   = open(os.path.join(pref,csp+".res_locisf_subsampled"),'w')

  for qid,idx in enumerate(indexes):
    labels  = [int(float(x))   for x in label_lines [idx].split("\n")[1:]]
    cnts    = [float(x) for x in cnts_lines  [idx].split("\n")[1:]]
    isfs    = [float(x) for x in isf_lines   [idx].split("\n")[1:]]
    idfs    = [float(x) for x in idf_lines   [idx].split("\n")[1:]]
    locisfs = [float(x) for x in locisf_lines[idx].split("\n")[1:]]
    for i,lab in enumerate(labels):
      label_out.write("%s 0 %s %d\n" % ( trail_id(qid+1),trail_id(i),lab) )
      cnt_out.write("%s\t0\t%s\t0\t%.1f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),cnts[i]))
      # wgt_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),wgts[i]))
      isf_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),isfs[i]))
      idf_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),idfs[i]))
      locisf_out.write("%s\t0\t%s\t0\t%.6f NEWSQA\n" % ( trail_id(qid+1),trail_id(i),locisfs[i]))
    ##
  ##
  label_out.close()
  cnt_out.close()
  isf_out.close()
  idf_out.close()
  locisf_out.close()
  
