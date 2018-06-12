####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pdb
import argparse
import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug

from data_utils import DataProcessor, BatchData
from my_flags import FLAGS
from my_model import MY_Model
from model_docsum import accuracy_qas_top
from train_test_utils import batch_predict_with_a_model, batch_load_data
from sklearn.model_selection import ParameterSampler, ParameterGrid
from scipy.stats.distributions import expon
from scipy.stats import lognorm, uniform

seed = 42

np.random.seed(seed)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Rutine for sweeping through hyper-parameters setups for the original sidenet')
  parser.add_argument('gpu',help='gpu id',type=str,default="0")
  parser.add_argument('dataset',help='Dataset to use / mode of FLAGS setup',type=str,default="newsqa")
  parser.add_argument('file_suffix',help='Suffix for exp name',type=str,default="")
  args = parser.parse_args()
  
  FLAGS.data_mode = args.dataset
  FLAGS.gpu_id = args.gpu
  FLAGS.use_subsampled_dataset = True
  FLAGS.force_reading = False
  FLAGS.train_dir = "/disk/ocean/rcardenas/sidenet_ems/train_dir_"+args.dataset+"_subs"
  FLAGS.train_epoch_crossentropy = 20

  FLAGS.max_sent_length = 50
  FLAGS.max_doc_length = 64
  FLAGS.max_filter_length = 8
  FLAGS.min_filter_length = 5
  FLAGS.batch_size = 128        # best, 256
  FLAGS.learning_rate = 0.0001   # best, 0.0001 they don't fit :(
  FLAGS.size = 600
  FLAGS.max_gradient_norm = 5 # best, periodic jumping fixed
  FLAGS.norm_extra_feats = True
  FLAGS.decorrelate_extra_feats = False

  FLAGS.sentembed_size = 256 # fix this
  FLAGS.use_dropout = False

  # set sentence, doc length to maximum

  output = open(FLAGS.data_mode + "_hp_grid_tuning_%s.txt" % args.file_suffix,'w')

  print("Reading vocabulary...")
  vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
  print("Reading training set...")
  train_data = DataProcessor().prepare_news_data(vocab_dict, data_type="training") # subsampled
  # data in whole batch with padded matrixes
  print("Reading validation set...")
  val_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict,
                                                                data_type="validation",
                                                                normalizer=train_data.normalizer,
                                                                pca_model=train_data.pca_model))

  setup_by_id = {}
  results_by_id = {}
  setup_id = 0
  best_global_acc = -1
  best_setup_id = -1

  parameter_grid = {
    "batch_size" : [128,64],
    #"learning_rate" : [0.0001]
    #"dropout": [0.5,0.65]
  }


  ## loop for hyperparams
  param_gen = ParameterGrid(parameter_grid)
  for setup in param_gen:
    setup_time = time.time()
    setup_by_id[setup_id] = setup
    
    #FLAGS.size = setup["size"]
    #FLAGS.sentembed_size = setup["sentembed_size"]
    FLAGS.batch_size = setup["batch_size"]
    #FLAGS.learning_rate = setup["learning_rate"]
    #FLAGS.dropout = setup["dropout"]
    
    # check if concat, then adjust sentemb size
    fil_lens_to_test = FLAGS.max_filter_length - FLAGS.min_filter_length + 1
    if FLAGS.handle_filter_output == "concat" and FLAGS.sentembed_size%fil_lens_to_test != 0:
      q = int(FLAGS.sentembed_size // fil_lens_to_test)
      FLAGS.sentembed_size = q * fil_lens_to_test
    setup["sentembed_size"] = FLAGS.sentembed_size # just for printing

    print("Setup ",setup_id,": ",setup)
    output.write("Setup %d: %s\n" % (setup_id,str(setup)))

    best_acc = -1
    best_ep = 0
    with tf.Graph().as_default() and tf.device('/gpu:'+FLAGS.gpu_id):
      config = tf.ConfigProto(allow_soft_placement = True)
      tf.set_random_seed(seed)
      with tf.Session(config = config) as sess:
        model = MY_Model(sess, len(vocab_dict)-2)
        init_epoch = 1
        sess.run(model.vocab_embed_variable.assign(word_embedding_array))
        
        for epoch in range(init_epoch, FLAGS.train_epoch_crossentropy+1):
          ep_time = time.time() # to check duration

          train_data.shuffle_fileindices()
          total_loss = 0
          # Start Batch Training
          step = 1
          while (step * FLAGS.batch_size) <= len(train_data.fileindices):
            # Get batch data as Numpy Arrays
            batch = train_data.get_batch(((step-1)*FLAGS.batch_size), (step * FLAGS.batch_size))

            # Run optimizer: optimize policy and reward estimator
            _,ce_loss = sess.run([model.train_op_policynet_withgold,
                                  model.cross_entropy_loss],
                                  feed_dict={model.document_placeholder: batch.docs,
                                             model.label_placeholder: batch.labels,
                                             model.weight_placeholder: batch.weights,
                                             model.isf_placeholder: batch.isf_score,
                                             model.idf_placeholder: batch.idf_score})
            total_loss += ce_loss
            # Increase step
            if step%500==0:
              print ("\tStep: ",step)
            step += 1
          #END-WHILE-TRAINING
          total_loss /= step
          FLAGS.authorise_gold_label = False
          # retrieve batch with updated logits in it
          val_batch = batch_predict_with_a_model(val_batch, "validation", model, session=sess)
          FLAGS.authorise_gold_label = True
          probs = sess.run(model.probs,feed_dict={model.logits_placeholder: val_batch.logits})
          validation_acc = accuracy_qas_top(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids)
          
          print("\tEpoch %2d || Train ce_loss: %4.3f || Val acc: %.4f || duration: %3.2f" % (epoch,total_loss,validation_acc,time.time()-ep_time))
          output.write("\tEpoch %2d || Train ce_loss: %4.3f || Val acc: %.4f || duration: %3.2f\n" % (epoch,total_loss,validation_acc,time.time()-ep_time))
          
          if validation_acc > best_acc:
            best_acc = validation_acc
            best_ep = epoch

          #break # for time testing
        #END-FOR-EPOCH
        results_by_id[setup_id] = (best_acc,best_ep)
        if best_acc > best_global_acc:
          best_global_acc = best_acc
          best_setup_id = setup_id
      # clear graph
      tf.reset_default_graph()
    #END-GRAPH
    
    print("Best result in this setup:",results_by_id[setup_id])
    print("Duration: %.4fsec" % (time.time()-setup_time))
    output.write("Best result in this setup: %.6f,%d\n" % (best_acc,best_ep))
    output.write("Duration: %.4fsec\n" % (time.time()-setup_time))
    setup_id += 1
  #END-FOR-PARAMS
  
  print("Best setup: ",setup_by_id[best_setup_id])
  print("  Acc: %.4f | Epoch: %d" % results_by_id[best_setup_id])
  output.write("Best setup: " + str(setup_by_id[best_setup_id]) + "\n")
  output.write("  Acc: %.4f | Epoch: %d\n" % results_by_id[best_setup_id])
  output.close()
