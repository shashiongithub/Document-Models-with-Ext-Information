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

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import math_ops

# from tf.nn import variable_scope
from my_flags import FLAGS
from model_utils import *
import pdb
from sklearn.metrics import average_precision_score as aps

np.random.seed(42)

### Various types of extractor

def sentence_extractor_nonseqrnn_noatt(sents_ext, encoder_state):
    """Implements Sentence Extractor: No attention and non-sequential RNN
    Args:
    sents_ext: Embedding of sentences to label for extraction
    encoder_state: encoder_state
    Returns:
    extractor output and logits
    """
    # Define Variables
    weight = variable_on_cpu('weight', [FLAGS.size, FLAGS.target_label_size], tf.random_normal_initializer())
    bias = variable_on_cpu('bias', [FLAGS.target_label_size], tf.random_normal_initializer())

    # Get RNN output
    rnn_extractor_output, _ = simple_rnn(sents_ext, initial_state=encoder_state)

    with variable_scope.variable_scope("Reshape-Out"):
        rnn_extractor_output =  reshape_list2tensor(rnn_extractor_output, FLAGS.max_doc_length, FLAGS.size)

        # Get Final logits without softmax
        extractor_output_forlogits = tf.reshape(rnn_extractor_output, [-1, FLAGS.size])
        logits = tf.matmul(extractor_output_forlogits, weight) + bias
        # logits: [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
        logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='sidenet-logits')
    return rnn_extractor_output, logits


def sentence_extractor_seqrnn_docatt(sents_ext, encoder_outputs, encoder_state, sents_labels):
    """Implements Sentence Extractor: Sequential RNN with attention over sentences during encoding
    Args:
    sents_ext: Embedding of sentences to label for extraction
    encoder_outputs, encoder_state
    sents_labels: Gold sent labels for training
    Returns:
    extractor output and logits
    """
    # Define MLP Variables
    weights = {
      'h1': variable_on_cpu('weight_1', [2*FLAGS.size, FLAGS.size], tf.random_normal_initializer()),
      'h2': variable_on_cpu('weight_2', [FLAGS.size, FLAGS.size], tf.random_normal_initializer()),
      'out': variable_on_cpu('weight_out', [FLAGS.size, FLAGS.target_label_size], tf.random_normal_initializer())
      }
    biases = {
      'b1': variable_on_cpu('bias_1', [FLAGS.size], tf.random_normal_initializer()),
      'b2': variable_on_cpu('bias_2', [FLAGS.size], tf.random_normal_initializer()),
      'out': variable_on_cpu('bias_out', [FLAGS.target_label_size], tf.random_normal_initializer())
      }

    # Shift sents_ext for RNN
    with variable_scope.variable_scope("Shift-SentExt"):
        # Create embeddings for special symbol (lets assume all 0) and put in the front by shifting by one
        special_tensor = tf.zeros_like(sents_ext[0]) #  tf.ones_like(sents_ext[0])
        sents_ext_shifted = [special_tensor] + sents_ext[:-1]

    # Reshape sents_labels for RNN (Only used for cross entropy training)
    with variable_scope.variable_scope("Reshape-Label"):
        # only used for training
        sents_labels = reshape_tensor2list(sents_labels, FLAGS.max_doc_length, FLAGS.target_label_size)

    # Define Sequential Decoder
    extractor_outputs, logits = jporg_attentional_seqrnn_decoder(sents_ext_shifted, encoder_outputs, encoder_state, sents_labels, weights, biases)

    # Final logits without softmax
    with variable_scope.variable_scope("Reshape-Out"):
        logits = reshape_list2tensor(logits, FLAGS.max_doc_length, FLAGS.target_label_size)
        extractor_outputs = reshape_list2tensor(extractor_outputs, FLAGS.max_doc_length, 2*FLAGS.size)

    return extractor_outputs, logits


### Training functions

def train_cross_entropy_loss(cross_entropy_loss):
    """ Training with Gold Label: Pretraining network to start with a better policy
    Args: cross_entropy_loss
    """
    with tf.variable_scope('TrainCrossEntropyLoss') as scope:

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='adam')

        # Compute gradients of policy network
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PolicyNetwork")
        grads_and_vars = optimizer.compute_gradients(cross_entropy_loss, var_list=policy_network_variables)
        grads_and_vars_capped_norm = grads_and_vars
        if FLAGS.max_gradient_norm != -1:
            grads_and_vars_capped_norm = [(tf.clip_by_norm(grad,FLAGS.max_gradient_norm), var) for grad, var in grads_and_vars]

        grads_to_summ = [tensor for tensor,var in grads_and_vars if tensor!=None]
        grads_to_summ = [tf.reshape(tensor,[-1]) for tensor in grads_to_summ 
                                                    if tensor.dtype==tf.float16 or 
                                                    tensor.dtype==tf.float32]
        grads_to_summ = tf.concat(0,grads_to_summ)
        # Apply Gradients
        return optimizer.apply_gradients(grads_and_vars_capped_norm),grads_to_summ

#####


### Accuracy Calculations

def accuracy(logits, labels, weights):
  """Estimate accuracy of predictions
  Args:
    logits: Logits from inference(). [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
  Returns:
    Accuracy: Estimates average of accuracy for each sentence
  """
  with tf.variable_scope('Accuracy') as scope:
    logits = tf.reshape(logits, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
    labels = tf.reshape(labels, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)) # [FLAGS.batch_size*FLAGS.max_doc_length]
    correct_pred =  tf.reshape(correct_pred, [-1, FLAGS.max_doc_length])  # [FLAGS.batch_size, FLAGS.max_doc_length]
    correct_pred = tf.cast(correct_pred, tf.float32)
    # Get Accuracy
    accuracy = tf.reduce_mean(correct_pred, name='accuracy')
    if FLAGS.weighted_loss:
      correct_pred = tf.mul(correct_pred, weights)
      correct_pred = tf.reduce_sum(correct_pred, reduction_indices=1) # [FLAGS.batch_size]
      doc_lengths = tf.reduce_sum(weights, reduction_indices=1) # [FLAGS.batch_size]
      correct_pred_avg = tf.div(correct_pred, doc_lengths)
      accuracy = tf.reduce_mean(correct_pred_avg, name='accuracy')
  return accuracy


def save_metrics(filename,idx,acc,mrr,_map):
  out = open(filename,"a")
  out.write("%d\t%.4f\t%.4f\t%.4f\n" % (idx,acc,mrr,_map))
  out.close()



### Accuracy QAS

def accuracy_qas_any(logits, labels, weights):
  """
  Estimate accuracy of predictions for Question Answering Selection
  If any sentence predicted as 1 is on the gold-sentences set (the answer set of sentences),
  then sample is correctly classified
  Args:
    logits: Logits from inference(). [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
  Returns:
    Accuracy: Estimates average of accuracy for each sentence
  """
  with tf.variable_scope('Accuracy_QAS_any') as scope:
    #logits = tf.reshape(logits, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
    #labels = tf.reshape(labels, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
    final_shape = tf.shape(weights)
    logits_oh = tf.equal(tf.argmax(logits,2),tf.zeros(final_shape,dtype=tf.int64))
    logits_oh = tf.cast(logits_oh, dtype=tf.float32)
    labels_oh = tf.equal(tf.argmax(labels,2),tf.zeros(final_shape,dtype=tf.int64))
    labels_oh = tf.cast(tf.argmax(labels,2), dtype=tf.float32) # [batch_size, max_doc_length]
    if FLAGS.weighted_loss:
      weights = tf.cast(weights,tf.float32)
      logits_oh = tf.mul(logits_oh,weights) # only need to mask one of two mats

    correct_pred = tf.reduce_sum(tf.mul(logits_oh,labels_oh),1) # [batch_size]
    #correct_pred = tf.diag_part(tf.matmul(logits_oh,labels_oh,transpose_b = True)) # [batch_size]
    correct_pred = tf.cast(correct_pred,dtype=tf.bool) # True if sum of matches is > 0
    correct_pred = tf.cast(correct_pred, tf.float32)
    # Get Accuracy
    accuracy = tf.reduce_mean(correct_pred, name='accuracy')

  return accuracy


### Accuracy QAS TOP-RANKED

def accuracy_qas_top(probs, labels, weights, scores):
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
  one_prob = probs[:,:,0]
  labels = labels[:,:,0]
  bs,ld = labels.shape

  if FLAGS.filtered_setting:
    # limit space search to top K ranked sents
    topk_mask = np.zeros([bs,ld],dtype=np.float32)
    for i in range(bs):
      for j in range(FLAGS.topK):
        if scores[i,j]==-1:
            break
        topk_mask[i,scores[i,j]] = 1.0
    one_prob = one_prob * topk_mask

  if FLAGS.weighted_loss:
    one_prob = one_prob * weights # only need to mask one of two mats

  mask = one_prob.sum(axis=1) > 0
  correct = 0.0
  for i in range(bs):
    if mask[i]:
      correct += labels[i,one_prob[i,:].argmax()] 
  
  accuracy = correct / bs

  return accuracy



def mrr_metric(probs,labels,weights,scores):
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

  if FLAGS.filtered_setting:
    # limit space search to top K ranked sents
    topk_mask = np.zeros([bs,ld],dtype=np.float32)
    for i in range(bs):
      for j in range(FLAGS.topK):
        if scores[i,j]==-1:
            break
        topk_mask[i,scores[i,j]] = 1.0
    one_prob = one_prob * topk_mask

  if FLAGS.weighted_loss:
    one_prob = one_prob * weights # only need to mask one of two mats

  max_gold_prob = np.max(one_prob*labels,axis=1) # maximum prob bw golden answer sentences acc by model [batch_size]
  mask = one_prob.sum(axis=1) > 0

  mrr = 0.0
  for id_doc in range(bs):
    if mask[id_doc] and max_gold_prob[id_doc]>0:
      rel_rank = 1 + sum(one_prob[id_doc,:]>max_gold_prob[id_doc]) # how many sentences have higher prob than most prob answer +1
      mrr += 1.0/rel_rank # accumulate inverse rank
  mrr = mrr / bs # mrr as mean of inverse rank

  return mrr



def map_score(probs,labels,weights,scores):
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
  one_prob = probs[:,:,0]
  bs,ld = one_prob.shape

  if FLAGS.filtered_setting:
    # limit space search to top K ranked sents
    topk_mask = np.zeros([bs,ld],dtype=np.float32)
    for i in range(bs):
      for j in range(FLAGS.topK):
        if scores[i,j]==-1:
            break
        topk_mask[i,scores[i,j]] = 1.0
    one_prob = one_prob * topk_mask

  if FLAGS.weighted_loss:
    one_prob = one_prob * weights # only need to mask one of two mats
    labels = labels * weights

  mask = one_prob.sum(axis=1) > 0
  aps_batch = 0.0
  for i in range(bs):
    if mask[i]:
      temp = np.zeros(ld)
      temp[one_prob[i,:].argmax()] = 1.0
      aps_batch += aps(temp,labels[i]) 
  
  map_sc = aps_batch / bs

  return map_sc

#####################
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
    aps_batch += aps(temp,labels[i]) 

  map_sc = aps_batch / bs

  return map_sc


