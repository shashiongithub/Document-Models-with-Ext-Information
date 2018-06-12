####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

import os, sys
from utils import DATA_PICKLE_DIR,PREPROC_DATA_DIR
import pdb

pref = os.path.join(PREPROC_DATA_DIR,"wikiqa")
for csp in ["training","validation","test"]:
  filtered_ref = ''
    
  label_lines = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".org_ent.label"),'r').read().strip("\n").split("\n\n")
  cnts_lines = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".cnt.scores"),'r').read().strip("\n").split("\n\n")
  wgt_lines = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".wgtcnt.scores"),'r').read().strip("\n").split("\n\n")

  idf_lines = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".idf.scores"),'r').read().strip("\n").split("\n\n")
  isf_lines = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".isf.scores"),'r').read().strip("\n").split("\n\n")
  locisf_lines = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".locisf.scores"),'r').read().strip("\n").split("\n\n")

  label_out = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".rel_info"),'w')
  cnt_out   = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".res_cnt"),'w')
  wgt_out   = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".res_wgt"),'w')
  isf_out   = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".res_isf"),'w')
  idf_out   = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".res_idf"),'w')
  locisf_out   = open(os.path.join(PREPROC_DATA_DIR,"wikiqa",csp+".res_locisf"),'w')
  
  for qid,labels in enumerate(label_lines):
    labels = labels.split("\n")
    labels = [int(x) for x in labels[1:]]
    cnts = [float(x) for x in cnts_lines[qid].split("\n")[1:]]
    wgts = [float(x) for x in wgt_lines[qid].split("\n")[1:]]
    isfs = [float(x) for x in isf_lines[qid].split("\n")[1:]]
    idfs = [float(x) for x in idf_lines[qid].split("\n")[1:]]
    locisfs = [float(x) for x in locisf_lines[qid].split("\n")[1:]]
    for i,lab in enumerate(labels):
      label_out.write("%d 0 %d %d\n" % (qid+1,i,lab) )
      try:
        cnt_out.write("%d\t0\t%d\t0\t%.1f WIKIQA\n" % (qid+1,i,cnts[i]))
        wgt_out.write("%d\t0\t%d\t0\t%.6f WIKIQA\n" % (qid+1,i,wgts[i]))
        isf_out.write("%d\t0\t%d\t0\t%.6f WIKIQA\n" % (qid+1,i,isfs[i]))
        idf_out.write("%d\t0\t%d\t0\t%.6f WIKIQA\n" % (qid+1,i,idfs[i]))
        locisf_out.write("%d\t0\t%d\t0\t%.6f WIKIQA\n" % (qid+1,i,locisfs[i]))
      except:
        pdb.set_trace()
    ##
  ##
  label_out.close()
  cnt_out.close()
  wgt_out.close()
  isf_out.close()
  idf_out.close()
  locisf_out.close()
  
