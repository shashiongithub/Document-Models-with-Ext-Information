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
from model_docsum import accuracy_qas_top, mrr_metric, group_by_doc
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
  
  FLAGS.force_reading = False
  #FLAGS.train_dir = "/disk/ocean/rcardenas/sidenet/train_dir_"+args.dataset
  FLAGS.train_dir = os.path.abspath("./train_dir_" + args.dataset+"_subs")
  FLAGS.train_epoch_crossentropy = 50

  if args.dataset=='wikiqa':
    FLAGS.use_subsampled_dataset = False
    FLAGS.max_sent_length =100 # WikiQA, ABCNN limit (40)
    FLAGS.max_doc_length = 30
  elif args.dataset=='newsqa':
    FLAGS.use_subsampled_dataset = True
    FLAGS.max_sent_length = 50 # , and ~95% of sentences until this thr
    FLAGS.max_doc_length = 64


  FLAGS.max_filter_length = 8 # 8
  FLAGS.min_filter_length = 5 # 5
  FLAGS.batch_size = 64        # best, 256
  FLAGS.learning_rate = 0.0001   # best, 0.0001 they don't fit :(
  FLAGS.max_gradient_norm = 15 # best, periodic jumping fixed
  
  FLAGS.sentembed_size = 348 # fix this
  FLAGS.mlp_size = 100

  FLAGS.use_dropout = False
  FLAGS.use_dropout_outatt = False
  FLAGS.dropout = 0.8

  prev_drpt = FLAGS.use_dropout


  # set sentence, doc length to maximum
  output = open("tunning_"+FLAGS.data_mode+"/"+FLAGS.data_mode + "_hp_grid_tuning_%s.txt" % args.file_suffix,'w')

  vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
  train_data = DataProcessor().prepare_news_data(vocab_dict, data_type="training") # subsampled
  # data in whole batch with padded matrixes
  val_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict, data_type="validation"))

  setup_by_id = {}
  results_by_id = {}
  results_by_id_mrr = {}
  setup_id = 0
  best_global_acc = -1
  best_global_mrr = -1
  best_setup_id = -1
  best_setup_id_mrr = -1

  parameter_grid = {
    "batch_size" : [64],
    "learning_rate" : [0.0001],
    "mlp_size":[100],
    "sentembed_size":[348],
    "use_dropout":[True,False],
    "dropout": [0.8],
    'max_gradient_norm': [10]
  }


  ## loop for hyperparams
  param_gen = ParameterGrid(parameter_grid)
  for setup in param_gen:
    setup_time = time.time()
    setup_by_id[setup_id] = setup
    #if setup["batch_size"] == 256 and setup["learning_rate"]>0.001:
    #  continue
    
    FLAGS.batch_size = setup["batch_size"]
    FLAGS.learning_rate = setup["learning_rate"]
    FLAGS.mlp_size = setup["mlp_size"]
    FLAGS.sentembed_size = setup["sentembed_size"] 
    FLAGS.use_dropout = setup["use_dropout"]
    FLAGS.dropout = setup["dropout"]
    FLAGS.max_gradient_norm = setup["max_gradient_norm"]
    prev_drpt = FLAGS.use_dropout
    
    # check if concat, then adjust sentemb size
    fil_lens_to_test = FLAGS.max_filter_length - FLAGS.min_filter_length + 1
    if FLAGS.handle_filter_output == "concat" and FLAGS.sentembed_size%fil_lens_to_test != 0:
      q = int(FLAGS.sentembed_size // fil_lens_to_test)
      FLAGS.sentembed_size = q * fil_lens_to_test
    setup["sentembed_size"] = FLAGS.sentembed_size # just for printing

    print("Setup ",setup_id,": ",setup)
    output.write("Setup %d: %s\n" % (setup_id,str(setup)))

    best_acc = -1
    best_mrr = -1
    best_ep = 0
    best_ep_mrr = 0
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
                                             model.label_placeholder: batch.labels})
            total_loss += ce_loss
            # Increase step
            if step%5000==0:
              print ("\tStep: ",step)
            step += 1
          #END-WHILE-TRAINING
          total_loss /= step
          FLAGS.authorise_gold_label = False
          FLAGS.use_dropout = False
          # retrieve batch with updated logits in it
          val_batch = batch_predict_with_a_model(val_batch, "validation", model, session=sess)
          FLAGS.authorise_gold_label = True
          FLAGS.use_dropout = prev_drpt
          probs = sess.run(model.predictions,feed_dict={model.logits_placeholder: val_batch.logits})
          probs,lab,w = group_by_doc(probs,val_batch.labels,val_batch.qids)
          validation_acc = accuracy_qas_top(probs,lab,w)
          val_mrr = mrr_metric(probs,lab,w,"validation")
 
          print("\tEpoch %2d || Train ce_loss: %4.3f || Val acc: %.4f || Val mrr: %.4f || duration: %3.2f" % 
            (epoch,total_loss,validation_acc,val_mrr,time.time()-ep_time))
          output.write("\tEpoch %2d || Train ce_loss: %4.3f || Val acc: %.4f || Val mrr: %.4f || duration: %3.2f\n" % 
            (epoch,total_loss,validation_acc,val_mrr,time.time()-ep_time))

          if validation_acc > best_acc:
            best_acc = validation_acc
            best_ep = epoch
          if val_mrr > best_mrr:
            best_mrr = val_mrr
            best_ep_mrr = epoch
          #break # for time testing
        #END-FOR-EPOCH
        results_by_id[setup_id] = (best_acc,best_ep)
        results_by_id_mrr[setup_id] = (best_mrr,best_ep_mrr)
        if best_acc > best_global_acc:
          best_global_acc = best_acc
          best_setup_id = setup_id
        if best_mrr > best_global_mrr:
          best_global_mrr = best_mrr
          best_setup_id_mrr = setup_id
      # clear graph
      tf.reset_default_graph()
    #END-GRAPH
    
    print("Best ACC result in this setup:",results_by_id[setup_id])
    print("Best MRR result in this setup:",results_by_id_mrr[setup_id])
    print("Duration: %.4fsec" % (time.time()-setup_time))
    output.write("Best acc result in this setup: %.6f,%d\n" % (best_acc,best_ep))
    output.write("Best mrr result in this setup: %.6f,%d\n" % (best_mrr,best_ep_mrr))
    output.write("Duration: %.4fsec\n" % (time.time()-setup_time))
    setup_id += 1
  #END-FOR-PARAMS
  
  print("Best acc setup: ",setup_by_id[best_setup_id])
  print("  Acc: %.4f | Epoch: %d" % results_by_id[best_setup_id])
  print("Best mrr setup: ",setup_by_id_mrr[best_setup_id_mrr])
  print("  MRR: %.4f | Epoch: %d" % results_by_id_mrr[best_setup_id_mrr])
  output.write("Best acc setup: " + str(setup_by_id[best_setup_id]) + "\n")
  output.write("  Acc: %.4f | Epoch: %d\n" % results_by_id[best_setup_id])
  output.write("Best mrr setup: " + str(setup_by_id[best_setup_id_mrr]) + "\n")
  output.write("  MRR: %.4f | Epoch: %d\n" % results_by_id_mrr[best_setup_id_mrr])
  output.close()
  
