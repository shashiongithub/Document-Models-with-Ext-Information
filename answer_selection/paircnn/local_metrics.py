####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project

# v1.2 XNET
#   author: Ronald Cardenas
####################################

"""
Question Answering Modules and Models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../../common')

import os
import numpy as np
import subprocess as sp
import tensorflow as tf
from my_flags import FLAGS
from sklearn.metrics import average_precision_score as aps

seed = 42
np.random.seed(seed)


##########
##########


def group_by_doc(probs,labels,qids):
    """
    probs : [batch_size x target_size]
    labels: [batch_size x target_size]
    qids  : [batch_size]
    """
    data_by_qid = {}
    bs = probs.shape[0]
    
    for idx in range(bs):
        qid = qids[idx]
        if qid not in data_by_qid:
            data_by_qid[qid] = []
        data_by_qid[qid].append( (probs[idx,0],labels[idx,0]) )
    n_qs = len(data_by_qid)
    prob_group   = np.zeros([n_qs,FLAGS.max_doc_length])
    labels_group = np.zeros([n_qs,FLAGS.max_doc_length])
    weights = np.zeros([n_qs,FLAGS.max_doc_length],dtype=int)

    for qid,p_l in data_by_qid.items():
        p = [x[0] for x in p_l]
        l = [x[1] for x in p_l]
        len_doc = len(p)
        
        zero_fill = [0]*(FLAGS.max_doc_length - len_doc)
        p = p + zero_fill
        l = l + zero_fill

        prob_group[qid-1,:]   = p[:]
        labels_group[qid-1,:] = l[:]
        weights[qid-1,:] = [1]*len_doc + zero_fill
    return prob_group,labels_group,weights


### Accuracy Calculations

def save_metrics(filename,idx,acc,mrr,_map):
  out = open(filename,"a")
  out.write("%d\t%.4f\t%.4f\t%.4f\n" % (idx,acc,mrr,_map))
  out.close()


### Accuracy QAS TOP-RANKED

# def accuracy_qas_top(one_prob, labels, weights):
#   """
#   Estimate accuracy of predictions for Question Answering Selection
#   If top-ranked sentence predicted as 1 is on the gold-sentences set (the answer set of sentences),
#   then sample is correctly classified
#   Args:
#     probs: Probabilities. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
#     labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
#     weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
#     scores: ISF score indexes sorted in reverse order [FLAGS.batch_size, FLAGS.topK]
#   Returns:
#     Accuracy: Estimates average of accuracy for each sentence
#   """
#   bs,ld = labels.shape

#   if FLAGS.weighted_loss:
#     one_prob = one_prob * weights # only need to mask one of two mats

#   mask = labels.sum(axis=1) > 0
#   correct = 0.0
#   total = 0.0
#   for i in range(bs):
#     if mask[i]==0 or mask[i]==sum(weights[i,:]):
#       continue
#     correct += labels[i,one_prob[i,:].argmax()] 
#     total += 1.0
#   accuracy = correct / total

#   return accuracy

def accuracy_qas_top(one_prob, labels, weights):
  """
  Estimate accuracy of predictions for Question Answering Selection
  If top-ranked sentence predicted as 1 is on the gold-sentences set (the answer set of sentences),
  then sample is correctly classified
  Args:
    probs: Probabilities. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
    scores: ISF score indexes sorted in reverse order [FLAGS.batch_size, FLAGS.topK]
  Returns:
    Accuracy: Estimates average of accuracy for each sentence
  """ 
  bs,ld = labels.shape

  if FLAGS.weighted_loss:
    one_prob = one_prob * weights # only need to mask one of two mats

  mask = labels.sum(axis=1) > 0
  correct = 0.0
  total = 0.0
  for i in range(bs):
    if mask[i]==0:
      continue

    if FLAGS.tie_break=="first":
      correct += labels[i,one_prob[i,:].argmax()]
    else:
      srt_ref = [(x,pos) for pos,x in enumerate(one_prob[i,:])]
      srt_ref.sort(reverse=True)
      correct += labels[i,srt_ref[0][1]]
    total += 1.0

  try:
    accuracy = correct / total if total!=0 else 0
  except:
    pdb.set_trace()

  return accuracy


def mrr_metric(one_prob,labels,weights,data_type):
  '''
  Calculates Mean reciprocal rank: mean(1/pos),
    pos : how many sentences are ranked higher than the answer-sentence with highst prob (given by model)
  Args:
    probs: Probabilities. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
  Returns:
    MRR: estimates MRR at document level
  '''

  bs,ld = one_prob.shape

  dump_trec_format(labels,one_prob,weights)
  popen = sp.Popen(["../trec_eval/trec_eval",
    "-m", "recip_rank",
    os.path.join(FLAGS.preprocessed_data_directory,FLAGS.data_mode,data_type+".rel_info"),
    os.path.join(FLAGS.train_dir,"temp.trec_res")],
    stdout=sp.PIPE)
  with popen.stdout as f:
    metric = f.read().strip("\n")[-6:]
    mrr = float(metric)
  return mrr



def map_score(one_prob,labels,weights,data_type):
  '''
  Calculates Mean Average Precision MAP
  Args:
    probs: Probabilities. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
  Returns:
    MAP: estimates MAP over all batch
  '''
  bs,ld = one_prob.shape

  if FLAGS.weighted_loss:
    one_prob = one_prob * weights # only need to mask one of two mats
    labels = labels * weights

  dump_trec_format(labels,one_prob,weights)
  popen = sp.Popen(["../trec_eval/trec_eval",
    "-m", "map",
    os.path.join(FLAGS.preprocessed_data_directory,FLAGS.data_mode,data_type+".rel_info"),
    os.path.join(FLAGS.train_dir,"temp.trec_res")],
    stdout=sp.PIPE)
  with popen.stdout as f:
    metric = f.read().strip("\n")[-6:]
    map_sc = float(metric)
  return map_sc


###############################################################

def trail_id(_id):
  sid = str(_id)
  sid = "0"*(7 - len(sid)) + sid
  return sid


def dump_trec_format(labels,scores,weights):
  bs,ld = labels.shape
  output = open(os.path.join(FLAGS.train_dir,"temp.trec_res"),'w')
  doc_lens = weights.sum(axis=1).astype(int)
  for qid in range(bs):
    for aid in range(doc_lens[qid]):
      output.write("%s 0 %s 0 %.6f 0\n" % (trail_id(qid+1),trail_id(aid),scores[qid,aid]))
  output.close()


###############################################################
def accuracy_qas_random(probs, labels, weights, scores):
  """
  Estimate accuracy of predictions for Question Answering Selection
  It takes random sentence as answer
  Args:
    probs: Probabilities. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
    scores: ISF score indexes sorted in reverse order [FLAGS.batch_size, FLAGS.topK]
  Returns:
    Accuracy: Estimates average of accuracy for each sentence
  """
  
  labels = labels[:,:,0]
  bs,ld = labels.shape

  if FLAGS.weighted_loss:
    labels = labels * weights # only need to mask one of two mats

  len_docs = weights.sum(axis=1)
  correct = 0.0
  for i in range(bs):
    rnd_idx = np.random.random_integers(0,len_docs[i]-1)
    correct += labels[i,rnd_idx] 
  
  accuracy = correct / bs

  return accuracy

def mrr_metric_random(probs,labels,weights,scores):
  '''
  Calculates Mean reciprocal rank: mean(1/pos),
    pos : how many sentences are ranked higher than the answer-sentence with highst prob (given by model)
  Args:
    probs: Probabilities. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
  Returns:
    MRR: estimates MRR at document level
  '''

  one_prob = probs[:,:,0] # slice prob of being 1 | [batch_size, max_doc_len]
  labels = labels[:,:,0] #[batch_size, max_doc_len]
  bs,ld = one_prob.shape

  if FLAGS.weighted_loss:
    one_prob = one_prob * weights # only need to mask one of two mats

  len_docs = weights.sum(axis=1)
  temp = np.zeros([bs,ld])
  for i in range(bs):
    rnd_idx = np.random.random_integers(0,len_docs[i]-1)
    temp[i,rnd_idx] = one_prob[i,rnd_idx]
  one_prob = temp

  max_gold_prob = np.max(one_prob*labels,axis=1) # maximum prob bw golden answer sentences acc by model [batch_size]
  mask = one_prob.sum(axis=1) > 0

  mrr = 0.0
  for id_doc in range(bs):
    if mask[id_doc]:
      rel_rank = 1 + sum(one_prob[id_doc,:]>max_gold_prob[id_doc]) # how many sentences have higher prob than most prob answer +1
      mrr += 1.0/rel_rank # accumulate inverse rank
  mrr = mrr / bs # mrr as mean of inverse rank

  return mrr


def map_score_random(probs,labels,weights,scores):
  '''
  Calculates Mean Average Precision MAP
  Args:
    probs: Probabilities. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
  Returns:
    MAP: estimates MAP over all batch
  '''
  labels = labels[:,:,0]
  bs,ld = labels.shape

  if FLAGS.weighted_loss:
    labels = labels * weights

  len_docs = weights.sum(axis=1)
  aps_batch = 0.0
  for i in range(bs):
    temp = np.zeros(ld)
    rnd_idx = np.random.random_integers(0,len_docs[i]-1)
    temp[rnd_idx] = 1.0
    aps_batch += aps(labels[i],temp) 

  map_sc = aps_batch / bs

  return map_sc


