####################################
# Author: Ronald Cardenas
# Date: July 2017
# Project: Document Modeling with External Attention for Sentence Extraction

####################################
import sys
sys.path.append('../../common')

from sigopt import Connection
import math
import os
import random
import sys
import time
import pdb
import argparse
import numpy as np
import tensorflow as tf

from data_utils import DataProcessor, BatchData
from my_flags import FLAGS
from my_model import MY_Model
from model_docsum import accuracy_qas_top, mrr_metric
from train_test_utils import batch_predict_with_a_model, batch_load_data

seed = 42
np.random.seed(seed)

#######################################################################################################



# Evaluate your model with the suggested parameter assignments
def evaluate_model(assignments,train_data,val_batch,score="mrr"):

  FLAGS.batch_size = assignments["batch_size"]
  FLAGS.learning_rate = math.exp(assignments["log_learning_rate"])
  FLAGS.size = assignments["size"]
  FLAGS.sentembed_size = assignments["sentembed_size"]

  #FLAGS.dropout = setup["dropout"]

  fil_lens_to_test = FLAGS.max_filter_length - FLAGS.min_filter_length + 1
  if FLAGS.handle_filter_output == "concat" and FLAGS.sentembed_size%fil_lens_to_test != 0:
    q = int(FLAGS.sentembed_size // fil_lens_to_test)
    FLAGS.sentembed_size = q * fil_lens_to_test

  print("Setup: bs: %d | lr: %f | size: %d | sent_emb: %d" % 
    (FLAGS.batch_size,
     FLAGS.learning_rate,
     FLAGS.size,
     FLAGS.sentembed_size)
    )

  with tf.Graph().as_default() and tf.device('/gpu:'+FLAGS.gpu_id):
    config = tf.ConfigProto(allow_soft_placement = True)
    tf.set_random_seed(seed)
    with tf.Session(config = config) as sess:
      model = MY_Model(sess, len(vocab_dict)-2)
      init_epoch = 1
      sess.run(model.vocab_embed_variable.assign(word_embedding_array))
      
      best_metric = -1
      best_ep = 0
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
                                           model.isf_score_placeholder: batch.isf_score,
                                           model.idf_score_placeholder: batch.idf_score,
                                           model.locisf_score_placeholder: batch.locisf_score})
          total_loss += ce_loss
          # Increase step
          if step%500==0:
            print ("\tStep: ",step)
          step += 1
        #END-WHILE-TRAINING
        total_loss /= step
        FLAGS.authorise_gold_label = False
        #FLAGS.use_dropout = False
        # retrieve batch with updated logits in it
        val_batch = batch_predict_with_a_model(val_batch, "validation", model, session=sess)
        FLAGS.authorise_gold_label = True
        #FLAGS.use_dropout = True
        probs = sess.run(model.predictions,feed_dict={model.logits_placeholder: val_batch.logits})
        if score=="acc":
          metric = accuracy_qas_top(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids)
        elif score=="mrr":
          metric = mrr_metric(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids,"validation")
        
        print("\tEpoch %2d || Train ce_loss: %4.3f || Val %s: %.4f || duration: %3.2f" % (epoch,total_loss,score,metric,time.time()-ep_time))
        
        if metric > best_metric:
          best_metric = metric
          best_ep = epoch
      #END-FOR-EPOCH
    # clear graph
    tf.reset_default_graph()
  #END-GRAPH
  print("Best metric:%.6f | ep: %d" % (best_metric,best_ep))

  return best_metric

#######################################################################################################

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Rutine for sweeping through hyper-parameters setups for the sidenet+')
  parser.add_argument('gpu',help='gpu id',type=str,default="0")
  parser.add_argument('dataset',help='Dataset to use / mode of FLAGS setup',type=str,default="newsqa")
  parser.add_argument('-acc','--acc', help='Use accuracy as metric',action="store_true")
  parser.add_argument('-mrr','--mrr', help='Use mrr as metric',action="store_true")
  args = parser.parse_args()
  
  FLAGS.data_mode = args.dataset if args.dataset!='squadnew' else 'squad'
  FLAGS.gpu_id = args.gpu
  FLAGS.train_dir = os.path.abspath("./train_dir_" + args.dataset+"_subs")
  FLAGS.train_epoch_crossentropy = 20

  if args.dataset=='wikiqa':
    FLAGS.use_subsampled_dataset = False
    FLAGS.max_sent_length =100 # WikiQA, ABCNN limit (40)
    FLAGS.max_doc_length = 30
  elif args.dataset=='newsqa':
    FLAGS.use_subsampled_dataset = True
    FLAGS.max_sent_length = 50 # , and ~95% of sentences until this thr
    FLAGS.max_doc_length = 64
  elif args.dataset=='squad' or args.dataset=='squadnew':
    FLAGS.use_subsampled_dataset = True
    FLAGS.max_sent_length = 80
    FLAGS.max_doc_length = 16
  elif args.dataset=='msmarco':
    FLAGS.use_subsampled_dataset = True
    FLAGS.max_sent_length = 150
    FLAGS.max_doc_length = 10

  FLAGS.pretrained_wordembedding_orgdata = os.path.expanduser("../datasets/word_emb/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec")
  FLAGS.preprocessed_data_directory = os.path.expanduser("../datasets/preprocessed_data")
  
  metric = 'mrr'
  if args.acc:
    metric = "acc"
  elif args.mrr:
    metric = "mrr"

  FLAGS.force_reading = False
  FLAGS.train_dir = os.path.abspath("./train_dir_" + args.dataset+"_subs")
  FLAGS.train_epoch_crossentropy = 20

  FLAGS.max_filter_length = 8 # 8
  FLAGS.min_filter_length = 5 # 5
  FLAGS.batch_size = 20        # best, 256
  FLAGS.learning_rate = 0.0001   # best, 0.0001 they don't fit :(
  FLAGS.size = 600
  FLAGS.max_gradient_norm = -1 #
  FLAGS.norm_extra_feats = True
  FLAGS.decorrelate_extra_feats = False
  FLAGS.sentembed_size = 350 #  348 fix this

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

  ###############################################################################################
  """
  # prueba
  # msmarco : 64 | _ | 2500 | 1024
  assignments = {
    'batch_size' : 64,
    'log_learning_rate' : -6.907755278982137,
    'size': 2500,
    'sentembed_size' : 1024
  }

  value = evaluate_model(assignments,train_data,val_batch,metric)  
  """
  
  conn = Connection(client_token="ICEQYJKIHWZSSICCIHLCZCSHNIYZQMPTRXRIRTNRSPOCCOJJ")

  # dict(name='dropout', type='double', bounds=dict(min=0.5, max=1.0)),

  experiment = conn.experiments().create(
      name='sidenet+ - %s - %s' %(FLAGS.data_mode,metric),
      parameters=[
          dict(name='batch_size'       , type='int'   , bounds=dict(min=20, max=64)),
          dict(name='log_learning_rate', type='double', bounds=dict(min=math.log(1e-6), max=math.log(1e-2))),
          dict(name='size'             , type='int'   , bounds=dict(min=100, max=2500)),
          dict(name='sentembed_size'   , type='int'   , bounds=dict(min=80, max=1024)),
      ],
  )
  print("Created experiment: https://sigopt.com/experiment/" + experiment.id)


  # Run the Optimization Loop between 10x - 20x the number of parameters
  for _ in range(100):
      suggestion = conn.experiments(experiment.id).suggestions().create()
      value = evaluate_model(suggestion.assignments,train_data,val_batch,metric)
      conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          value=value,
      )
  
