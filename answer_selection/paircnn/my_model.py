####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project
# Comments: Jan 2017
# Improved for Reinforcement Learning
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


import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import model_docsum
from my_flags import FLAGS
import model_utils

###################### Define Final Network ############################

class MY_Model:
  def __init__(self, sess, vocab_size):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

    ### Few variables that has been initianlised here
    # Word embedding variable
    self.vocab_embed_variable = model_utils.get_vocab_embed_variable(vocab_size)

    ### Define Place Holders
    self.document_placeholder = tf.placeholder("int32", [None,2,FLAGS.max_sent_length], name='doc-ph')
    self.label_placeholder = tf.placeholder(dtype, [None, FLAGS.target_label_size], name='label-ph')

    self.train_acc_placeholder = tf.placeholder(tf.float32,name='tacc')
    self.val_acc_placeholder   = tf.placeholder(tf.float32,name='vacc')

    # Only used for test/validation corpus
    self.logits_placeholder = tf.placeholder(dtype, [None, FLAGS.target_label_size], name='logits-ph')

    ### Define Ablated Network: Consists only Convolution.
    self.logits = model_docsum.policy_network_paircnn_qa(
                                              self.vocab_embed_variable,
                                              self.document_placeholder)

    ### Define Supervised Cross Entropy Loss
    self.cross_entropy_loss = model_docsum.cross_entropy_loss_paircnn_qa(self.logits, self.label_placeholder)
    self.cross_entropy_loss_val = model_docsum.cross_entropy_loss_paircnn_qa(self.logits_placeholder,self.label_placeholder)

    ### Define training operators & gradients
    self.train_op_policynet_withgold, self.gradients = model_docsum.train_cross_entropy_loss(self.cross_entropy_loss)

    self.predictions = model_utils.convert_logits_to_softmax_paircnn(self.logits_placeholder,None,False)

    # Create a saver.
    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

    # Scalar Summary Operations
    # Scalar Summary Operations
    _ = tf.scalar_summary("cross-entropy-loss", self.cross_entropy_loss)
    _ = tf.histogram_summary("gradients",self.gradients)
    self.merged = tf.merge_all_summaries()
    
    self.ce_loss_summary_val = tf.scalar_summary("cross-entropy-loss-validation", self.cross_entropy_loss_val)
    self.tstepa_accuracy_summary = tf.scalar_summary("training_accuracy_stepa", self.train_acc_placeholder)
    self.vstepa_accuracy_summary = tf.scalar_summary("validation_accuracy_stepa", self.val_acc_placeholder)
    
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    # Start running operations on the Graph.
    sess.run(init)    

    # Create Summary Graph for Tensorboard
    self.summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
