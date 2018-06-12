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
from sklearn.metrics import average_precision_score as aps
import subprocess as sp
import os
import pdb

seed = 42
np.random.seed(seed)

### Various types of extractor

def policy_network(vocab_embed_variable, document_placeholder):
    """Build the policy core network.
    Args:
    vocab_embed_variable: [vocab_size, FLAGS.wordembed_size], embeddings without PAD and UNK
    document_placeholder: [None,(FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                 FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.max_sent_length]
    label_placeholder: Gold label [None, FLAGS.max_doc_length, FLAGS.target_label_size], only used during cross entropy training of JP's model.
    isf_scores: ISF scores per sentence [None, FLAGS.max_doc_length]
    Returns:
    Outputs of sentence extractor and logits without softmax
    """

    with tf.variable_scope('CNN_baseline') as scope:

        ### Full Word embedding Lookup Variable
        # PADDING embedding non-trainable
        pad_embed_variable = variable_on_cpu("pad_embed", [1, FLAGS.wordembed_size], tf.constant_initializer(0), trainable=False)
        # UNK embedding trainable
        unk_embed_variable = variable_on_cpu("unk_embed", [1, FLAGS.wordembed_size], tf.constant_initializer(0), trainable=True)
        # Get fullvocab_embed_variable
        fullvocab_embed_variable = tf.concat(0, [pad_embed_variable, unk_embed_variable, vocab_embed_variable])
        # print(fullvocab_embed_variable)

        ### Lookup layer
        with tf.variable_scope('Lookup') as scope:
            document_placeholder_flat = tf.reshape(document_placeholder, [-1])
            document_word_embedding = tf.nn.embedding_lookup(fullvocab_embed_variable, document_placeholder_flat, name="Lookup")
            document_word_embedding = tf.reshape(document_word_embedding, [-1, 2,FLAGS.max_sent_length, FLAGS.wordembed_size])
            # print(document_word_embedding)

        ### Convolution Layer
        with tf.variable_scope('ConvLayer') as scope:
            document_word_embedding = tf.reshape(document_word_embedding, [-1, FLAGS.max_sent_length, FLAGS.wordembed_size])
            document_sent_embedding = conv1d_layer_sentence_representation(document_word_embedding) # [None, sentembed_size]
            document_sent_embedding = tf.reshape(document_sent_embedding, [-1, 2, FLAGS.sentembed_size])

        ### Reshape Tensor to List [-1, 2, sentembed_size] -> List of [-1, sentembed_size]
        with variable_scope.variable_scope("ReshapeDoc_TensorToList"):
            document_sent_embedding = reshape_tensor2list(document_sent_embedding, 2, FLAGS.sentembed_size)


        with variable_scope.variable_scope("Composing_MLP"):
            candidate = document_sent_embedding[0]
            question  = document_sent_embedding[1]
            composed = tf.concat(1,[candidate,question])

            in_prob = FLAGS.dropout if FLAGS.use_dropout else 1.0
            composed = tf.nn.dropout(composed,keep_prob=in_prob,seed=seed)

            weight = variable_on_cpu('weight_ff', [2*FLAGS.sentembed_size, FLAGS.mlp_size], tf.random_normal_initializer(seed=seed))
            bias = variable_on_cpu('bias_ff', [FLAGS.mlp_size], tf.random_normal_initializer(seed=seed))
            weight_out = variable_on_cpu('weight_out', [FLAGS.mlp_size, FLAGS.target_label_size], tf.random_normal_initializer(seed=seed))
            bias_out = variable_on_cpu('bias_out', [FLAGS.target_label_size], tf.random_normal_initializer(seed=seed))

            mlp = tf.nn.relu(tf.matmul(composed, weight) + bias,name='ff_layer')
            logits = tf.matmul(mlp, weight_out) + bias_out

    return logits


def cross_entropy_loss(logits, labels):
    """Estimate cost of predictions
    Add summary for "cost" and "cost/avg".
    Args:
    logits: Logits from inference(). [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
    Returns:
    Cross-entropy Cost
    """
    with tf.variable_scope('CrossEntropyLoss') as scope:
        # Reshape logits and labels to match the requirement of softmax_cross_entropy_with_logits
        #logits = tf.reshape(logits, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size, FLAGS.target_label_size]
        #labels = tf.reshape(labels, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size, FLAGS.target_label_size]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels) # [FLAGS.batch_size]
        #cross_entropy = tf.reshape(cross_entropy, [-1, FLAGS.max_doc_length])  # [FLAGS.batch_size, FLAGS.max_doc_length]


        # Cross entroy / document
        #cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) # [FLAGS.batch_size]
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='crossentropy')

        tf.add_to_collection('cross_entropy_loss', cross_entropy_mean)

    return cross_entropy_mean

### Training functions

def train_cross_entropy_loss(cross_entropy_loss):
    """ Training with Gold Label: Pretraining network to start with a better policy
    Args: cross_entropy_loss
    """
    with tf.variable_scope('TrainCrossEntropyLoss') as scope:

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='adam')

        # Compute gradients of policy network
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="CNN_baseline")
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
    if mask[i]==0 or mask[i]==sum(weights[i,:]):
      continue
    correct += labels[i,one_prob[i,:].argmax()] 
    total += 1.0
  accuracy = correct / total

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
def dump_trec_format(labels,scores,weights):
  bs,ld = labels.shape
  output = open(os.path.join(FLAGS.train_dir,"temp.trec_res"),'w')
  doc_lens = weights.sum(axis=1).astype(int)
  for qid in range(bs):
    for aid in range(doc_lens[qid]):
      output.write("%d 0 %d 0 %.6f 0\n" % (qid+1,aid,scores[qid,aid]))
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


