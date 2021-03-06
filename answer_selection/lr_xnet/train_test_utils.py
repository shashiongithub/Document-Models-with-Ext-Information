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
import pdb
import numpy as np
import tensorflow as tf

from model_docsum import mrr_metric, map_score, accuracy_qas_top, accuracy_qas_any, save_metrics
from data_utils import DataProcessor, BatchData
from my_model import MY_Model
from my_flags import FLAGS


def batch_load_data(data):
  main_batch = BatchData(None,None,None,None,None,None,None,None,None,None)
  step = 1
  while (step * FLAGS.batch_size) <= len(data.fileindices):
    # Get batch data as Numpy Arrays : Without shuffling
    batch = data.get_batch(((step-1)*FLAGS.batch_size), (step * FLAGS.batch_size))
    main_batch.extend(batch)
    # Increase step
    step += 1
  # Check if any data left
  if (len(data.fileindices) > ((step-1)*FLAGS.batch_size)):
    # Get last batch as Numpy Arrays
    batch = data.get_batch(((step-1)*FLAGS.batch_size), len(data.fileindices))
    main_batch.extend(batch)

  # Convert list to tensors
  main_batch.concat_batches()
  
  return main_batch


def batch_predict_with_a_model(batch,data_type,model, session=None):
  step = 1
  cos_sim_list = []
  logits_list = []
  while (step * FLAGS.batch_size) <= len(batch.docs):
    # Get batch data as Numpy Arrays : Without shuffling
    start_idx = (step-1)*FLAGS.batch_size
    end_idx = step * FLAGS.batch_size # not inclusive
    docs = batch.docs[start_idx:end_idx]
    labels = batch.labels[start_idx:end_idx]
    isf_score = batch.isf_score[start_idx:end_idx]
    locisf_score = batch.locisf_score[start_idx:end_idx]
    idf_score = batch.idf_score[start_idx:end_idx]
    
    if FLAGS.load_prediction==-1:
      batch_logits = session.run(model.logits,feed_dict={model.document_placeholder: docs,
                                                   model.label_placeholder: labels,
                                                   model.isf_score_placeholder: isf_score,
                                                   model.locisf_score_placeholder: locisf_score,
                                                   model.idf_score_placeholder: idf_score})
      logits_list.append(batch_logits)
    # Increase step
    step += 1

  # Check if any data left
  if (len(batch.docs) > ((step-1)*FLAGS.batch_size)):
    # Get last batch as Numpy Arrays
    start_idx = (step-1)*FLAGS.batch_size
    end_idx = len(batch.docs) # not inclusive
    docs = batch.docs[start_idx:end_idx]
    labels = batch.labels[start_idx:end_idx]
    isf_score = batch.isf_score[start_idx:end_idx]
    locisf_score = batch.locisf_score[start_idx:end_idx]
    idf_score = batch.idf_score[start_idx:end_idx]
    
    if FLAGS.load_prediction==-1:
      batch_logits = session.run(model.logits,feed_dict={model.document_placeholder: docs,
                                                   model.label_placeholder: labels,
                                                   model.isf_score_placeholder: isf_score,
                                                   model.locisf_score_placeholder: locisf_score,
                                                   model.idf_score_placeholder: idf_score})
      logits_list.append(batch_logits)
  
  if FLAGS.load_prediction!=-1:
    print("Loading netword predictions and embeddings...")
    fn_logits = "step-a.model.ckpt.epoch-%d.%s-prediction" % (FLAGS.load_prediction,data_type)
    #fn_cos_sim = "step-a.model.ckpt.epoch-%d-cos_sim.%s" % (FLAGS.load_prediction,data_type)
    
    logits_list = data.load_prediction(fn_logits)
    #cos_sim_list = data.load_prediction(fn_cos_sim)
  else:
    # Concatenate logits and cos_sim
    logits_list = np.vstack(logits_list)
    #cos_sim_list = np.vstack(cos_sim_list)
  batch.logits = logits_list
  #batch.cos_sim = cos_sim_list
  return batch


######################## CPU/GPU conf functions ###########################


def meta_experiment_gpu_conf(mode):
  # Training: use the tf default graph
  with tf.Graph().as_default() and tf.device('/gpu:'+FLAGS.gpu_id):

    config = tf.ConfigProto(allow_soft_placement = True)

    # Start a session
    with tf.Session(config = config) as sess:
      if mode=='train':
        train(sess)
      elif mode=='test':
        test(sess)
      elif mode=='test_train':
        test_train(sess)
      elif mode=='test_val':
        test_val(sess)
      elif mode=='train_debug':
        train_debug(sess)


def meta_experiment_cpu_conf(mode):
  # Training: use the tf default graph
  with tf.Graph().as_default():
    # Start a session
    with tf.Session() as sess:
      if mode=='train':
        train(sess)
      elif mode=='test':
        test(sess)
      elif mode=='test_train':
        test_train(sess)
      else:
        test_val(sess)

# ######################## Train Mode ###########################


def train(sess):
  """
  Training Mode: Create a new model and train the network
  """
  ## wrap debugger around session

  ### Prepare data for training
  print("Prepare vocab dict and read pretrained word embeddings ...")
  vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
  # vocab_dict contains _PAD and _UNK but not word_embedding_array

  print("Prepare training data ...")
  train_data = DataProcessor().prepare_news_data(vocab_dict, data_type="training")

  print("Prepare validation data ...")
  # data in whole batch with padded matrixes
  val_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict, data_type="validation"))

  # Create Model with various operations
  model = MY_Model(sess, len(vocab_dict)-2)
  
  init_epoch = 1
  # Resume training if indicated Select the model
  if FLAGS.model_to_load!=-1:
    if (os.path.isfile(FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load))):
      selected_modelpath = FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load)
    else:
      print("Model not found in checkpoint folder.")
      exit(0)

    # Reload saved model and test
    init_epoch = FLAGS.model_to_load + 1
    print("Reading model parameters from %s" % selected_modelpath)
    model.saver.restore(sess, selected_modelpath)
    print("Model loaded.")

  # Initialize word embedding before training
  print("Initialize word embedding vocabulary with pretrained embeddings ...")
  sess.run(model.vocab_embed_variable.assign(word_embedding_array))

  ### STEP A : Start Pretraining the policy with Supervised Labels: Simple Cross Entropy Training

  for epoch in range(init_epoch, FLAGS.train_epoch_crossentropy+1):
    ep_time = time.time() # to check duration

    print("STEP A: Epoch "+str(epoch)+" : Start pretraining with supervised labels")

    print("STEP A: Epoch "+str(epoch)+" : Reshuffle training document indices")
    train_data.shuffle_fileindices()

    # Start Batch Training
    step = 1
    while (step * FLAGS.batch_size) <= len(train_data.fileindices):
      # Get batch data as Numpy Arrays
      batch = train_data.get_batch(((step-1)*FLAGS.batch_size), (step * FLAGS.batch_size))
      # Run optimizer: optimize policy and reward estimator
      _,batch_logits,ce_loss,merged_summ = sess.run([
                                model.train_op_policynet_withgold,
                                model.logits,
                                model.cross_entropy_loss,
                                model.merged],
                                feed_dict={model.document_placeholder: batch.docs,
                                           model.label_placeholder: batch.labels,
                                           model.weight_placeholder: batch.weights,
                                           #model.cnt_placeholder: batch.cnt_score,
                                           model.isf_placeholder: batch.isf_score,
                                           model.idf_placeholder: batch.idf_score,
                                           model.locisf_score_placeholder:  batch.locisf_score
                                           #model.sent_len_placeholder: batch.sent_lens

                                           })

      # Print the progress
      if (step % FLAGS.training_checkpoint) == 0:
        probs = sess.run(model.probs,feed_dict={model.logits_placeholder: batch_logits})
        training_acc = accuracy_qas_top( probs,
                                         batch.labels,
                                         batch.weights,
                                         batch.isf_score_ids)
        acc_sum = sess.run( model.tstepa_accuracy_summary,
                            feed_dict={model.train_acc_placeholder: training_acc})

        print("STEP A: Epoch "+str(epoch)+" : Covered " + str(step*FLAGS.batch_size)+"/"+str(len(train_data.fileindices))+
              " : Minibatch CE Loss= {:.6f}".format(ce_loss) + ", Minibatch Accuracy= {:.6f}".format(training_acc))

        # Print Summary to Tensor Board
        model.summary_writer.add_summary(merged_summ, (epoch-1)*len(train_data.fileindices)+step*FLAGS.batch_size)
        model.summary_writer.add_summary(acc_sum, (epoch-1)*len(train_data.fileindices)+step*FLAGS.batch_size)

      # Increase step
      step += 1
      #break
    #END-WHILE-TRAINING

    # Save Model
    print("STEP A: Epoch "+str(epoch)+" : Saving model after epoch completion")
    checkpoint_path = os.path.join(FLAGS.train_dir, "step-a.model.ckpt.epoch-"+str(epoch))
    model.saver.save(sess, checkpoint_path)

    # Performance on the validation set
    print("STEP A: Epoch "+str(epoch)+" : Performance on the validation data")
    # Get Predictions: Prohibit the use of gold labels
    FLAGS.authorise_gold_label = False
    FLAGS.use_dropout = False
    val_batch = batch_predict_with_a_model(val_batch,"validation", model, session=sess)
    FLAGS.use_dropout = True
    FLAGS.authorise_gold_label = True

    # Validation Accuracy and Prediction
    probs = sess.run(model.predictions,feed_dict={model.logits_placeholder: val_batch.logits})
    validation_acc = accuracy_qas_top(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids)
    ce_loss_val, ce_loss_sum, acc_sum = sess.run([ model.cross_entropy_loss_val,
                                                   model.ce_loss_summary_val,
                                                   model.vstepa_accuracy_summary],
                                                  feed_dict={model.logits_placeholder: val_batch.logits,
                                                             model.label_placeholder:  val_batch.labels,
                                                             model.weight_placeholder: val_batch.weights,
                                                             model.train_acc_placeholder: validation_acc})

    # Print Validation Summary
    model.summary_writer.add_summary(acc_sum, epoch*len(train_data.fileindices))
    model.summary_writer.add_summary(ce_loss_sum, epoch*len(train_data.fileindices))

    print("STEP A: Epoch %s : Validation (%s) CE error = %.6f" % (str(epoch),str(val_batch.docs.shape[0]),ce_loss_val))
    print("STEP A: Epoch %s : Validation (%s) Accuracy = %.6f" % (str(epoch),str(val_batch.docs.shape[0]),validation_acc))
    # Estimate MRR on validation set
    mrr_score = mrr_metric(probs,val_batch.labels,val_batch.weights,val_batch.isf_score_ids)
    print("STEP A: Epoch %s : Validation (%s) MRR = %.6f" % (str(epoch),str(val_batch.docs.shape[0]),mrr_score))
    # Estimate MAP score on validation set
    mapsc = map_score(probs,val_batch.labels,val_batch.weights,val_batch.isf_score_ids)
    print("STEP A: Epoch %s : Validation (%s) MAP = %.6f" % (str(epoch),str(val_batch.docs.shape[0]),mapsc))

    print("STEP A: Epoch %d : Duration: %.4f" % (epoch,time.time()-ep_time) )
  #END-FOR-EPOCH

  print("Optimization Finished!")



def train_debug(sess):
  """
  Training Mode: Create a new model and train the network
  """
  ### Prepare data for training
  print("Prepare vocab dict and read pretrained word embeddings ...")
  vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
  # vocab_dict contains _PAD and _UNK but not word_embedding_array

  print("Prepare training data ...")
  train_data = DataProcessor().prepare_news_data(vocab_dict, data_type="training")
  print("Training size: ",len(train_data.fileindices))

  print("Prepare validation data ...")
  # data in whole batch with padded matrixes
  val_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict,
                                                                data_type="validation",
                                                                normalizer=train_data.normalizer,
                                                                pca_model=train_data.pca_model))
  print("Validation size: ",val_batch.docs.shape[0])

  #print("Prepare ROUGE reward generator ...")
  #rouge_generator = Reward_Generator()

  # Create Model with various operations
  model = MY_Model(sess, len(vocab_dict)-2)
  
  init_epoch = 1
  # Resume training if indicated Select the model

  # Initialize word embedding before training
  print("Initialize word embedding vocabulary with pretrained embeddings ...")
  sess.run(model.vocab_embed_variable.assign(word_embedding_array))

  ### STEP A : Start Pretraining the policy with Supervised Labels: Simple Cross Entropy Training
  counter = 0
  for epoch in range(init_epoch, FLAGS.train_epoch_crossentropy+1):
    ep_time = time.time() # to check duration

    train_data.shuffle_fileindices()

    # Start Batch Training
    step = 1
    total_ce_loss = 0
    total_train_acc = 0
    while (step * FLAGS.batch_size) <= len(train_data.fileindices):
      # Get batch data as Numpy Arrays
      batch = train_data.get_batch(((step-1)*FLAGS.batch_size), (step * FLAGS.batch_size))

      # Run optimizer: optimize policy and reward estimator
      _,batch_logits,ce_loss,merged_summ = sess.run([
                                model.train_op_policynet_withgold,
                                model.logits,
                                model.cross_entropy_loss,
                                model.merged],
                                feed_dict={model.document_placeholder: batch.docs,
                                           model.label_placeholder: batch.labels,
                                           model.weight_placeholder: batch.weights,
                                           #model.cnt_placeholder: batch.cnt_score,
                                           model.isf_placeholder: batch.isf_score,
                                           model.idf_placeholder: batch.idf_score,
                                           model.locisf_score_placeholder:  batch.locisf_score
                                           #model.sent_len_placeholder: batch.sent_lens
                                           })

      total_ce_loss += ce_loss
      probs = sess.run(model.predictions,feed_dict={model.logits_placeholder: batch_logits})
      training_acc = accuracy_qas_top( probs,
                                       batch.labels,
                                       batch.weights,
                                       batch.isf_score_ids)
      total_train_acc += training_acc
      # Print the progress
      if (step % FLAGS.training_checkpoint) == 0:
        total_train_acc /= FLAGS.training_checkpoint
        acc_sum = sess.run( model.tstepa_accuracy_summary,
                            feed_dict={model.train_acc_placeholder: training_acc})
        
        total_ce_loss /= FLAGS.training_checkpoint
        # Print Summary to Tensor Board
        model.summary_writer.add_summary(merged_summ, counter)
        model.summary_writer.add_summary(acc_sum, counter)

        # Performance on the validation set
        # Get Predictions: Prohibit the use of gold labels
        FLAGS.authorise_gold_label = False
        FLAGS.use_dropout = False
        val_batch = batch_predict_with_a_model(val_batch,"validation", model, session=sess)
        FLAGS.authorise_gold_label = True
        FLAGS.use_dropout = True

        # Validation Accuracy and Prediction
        probs = sess.run(model.probs,feed_dict={model.logits_placeholder: val_batch.logits})
        validation_acc = accuracy_qas_top(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids)

        ce_loss_val, ce_loss_sum, acc_sum = sess.run([ model.cross_entropy_loss_val,
                                                       model.ce_loss_summary_val,
                                                       model.vstepa_accuracy_summary],
                                                      feed_dict={model.logits_placeholder: val_batch.logits,
                                                                 model.label_placeholder:  val_batch.labels,
                                                                 model.weight_placeholder: val_batch.weights,
                                                                 model.isf_score_placeholder:  val_batch.isf_score,
                                                                 model.locisf_score_placeholder:  val_batch.locisf_score,
                                                                 model.val_acc_placeholder: validation_acc})

        # Print Validation Summary
        model.summary_writer.add_summary(acc_sum, counter)
        model.summary_writer.add_summary(ce_loss_sum, counter)
        
        print("Epoch %2d, step: %2d(%2d) || CE loss || Train : %4.3f , Val : %4.3f ||| ACC || Train : %.3f , Val : %.3f" % 
            (epoch,step,counter,total_ce_loss,ce_loss_val,total_train_acc,validation_acc))
        total_ce_loss = 0
        total_train_acc = 0

      if (step % 5) == 0: # to have comparable tensorboard plots
        counter += 1

      # Increase step
      step += 1
      
    #END-WHILE-TRAINING  ... but wait there is more 
    print("Epoch %2d : Duration: %.4f" % (epoch,time.time()-ep_time) )
    if not FLAGS.use_subsampled_dataset:
      print("Saving model after epoch completion")
      checkpoint_path = os.path.join(FLAGS.train_dir, "step-a.model.ckpt.epoch-"+str(epoch))
      model.saver.save(sess, checkpoint_path)

  #END-FOR-EPOCH

  print("Optimization Finished!")




# ######################## Test Mode ###########################

def test(sess):
  ### Prepare data for training
  print("Prepare vocab dict and read pretrained word embeddings ...")
  vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
  # vocab_dict contains _PAD and _UNK but not word_embedding_array

  print("Prepare test data ...")
  train_data = DataProcessor().prepare_news_data(vocab_dict, data_type="training")
  test_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict,
                                                                data_type="test",
                                                                normalizer=train_data.normalizer,
                                                                pca_model=train_data.pca_model))
  del train_data

  # Create Model with various operations
  model = MY_Model(sess, len(vocab_dict)-2)

  # # Initialize word embedding before training
  #print("Initialize word embedding vocabulary with pretrained embeddings ...")
  #sess.run(model.vocab_embed_variable.assign(word_embedding_array))

  # Select the model
  if (os.path.isfile(FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load))):
    selected_modelpath = FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load)
  else:
    print("Model not found in checkpoint folder.")
    exit(0)

  # Reload saved model and test
  print("Reading model parameters from %s" % selected_modelpath)
  model.saver.restore(sess, selected_modelpath)
  print("Model loaded.")

  # Initialize word embedding before training
  print("Initialize word embedding vocabulary with pretrained embeddings ...")
  sess.run(model.vocab_embed_variable.assign(word_embedding_array))

  # Test Accuracy and Prediction
  print("Performance on the test data:")
  FLAGS.authorise_gold_label = False
  test_batch = batch_predict_with_a_model(test_batch,"test",model, session=sess)
  probs = sess.run(model.probs,feed_dict={model.logits_placeholder: test_batch.logits})

  test_acc = accuracy_qas_top(probs, test_batch.labels, test_batch.weights, test_batch.isf_score_ids)

  # Print Test Summary
  print("Test ("+str(test_batch.docs.shape[0])+") Accuracy = {:.6f}".format(test_acc))
  # Estimate MRR on validation set
  mrr_score = mrr_metric(probs, test_batch.labels, test_batch.weights, test_batch.isf_score_ids)
  print("Test ("+str(test_batch.docs.shape[0])+") MRR = {:.6f}".format(mrr_score))
  # Estimate MAP score on validation set
  mapsc = map_score(probs, test_batch.labels, test_batch.weights, test_batch.isf_score_ids)
  print("Test ("+str(test_batch.docs.shape[0])+") MAP = {:.6f}".format(mapsc))

  # Writing test predictions and final summaries
  modelname = "step-a.model.ckpt.epoch-" + str(FLAGS.model_to_load)
  write_prediction_summaries(test_batch, probs, modelname, "test")



# ######################## Test Mode on Training Data ###########################

def test_train(sess):
  """
  Test Mode: Loads an existing model and test it on the training set
  """
  ### Prepare data for training
  print("Prepare vocab dict and read pretrained word embeddings ...")
  vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
  # vocab_dict contains _PAD and _UNK but not word_embedding_array

  print("Prepare test data ...")
  test_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict, data_type="training"))

  # Create Model with various operations
  model = MY_Model(sess, len(vocab_dict)-2)

  # Select the model
  if (os.path.isfile(FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load))):
    selected_modelpath = FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load)
  else:
    print("Model not found in checkpoint folder.")
    exit(0)

  # Reload saved model and test
  print("Reading model parameters from %s" % selected_modelpath)
  model.saver.restore(sess, selected_modelpath)
  print("Model loaded.")

  # Initialize word embedding before training
  print("Initialize word embedding vocabulary with pretrained embeddings ...")
  sess.run(model.vocab_embed_variable.assign(word_embedding_array))

  # Test Accuracy and Prediction
  print("Performance on the training data:")
  FLAGS.authorise_gold_label = False
  test_batch = batch_predict_with_a_model(test_batch,"training",model, session=sess)
  probs = sess.run(model.probs,feed_dict={model.logits_placeholder: test_batch.logits})

  test_acc = accuracy_qas_top(probs, test_batch.labels, test_batch.weights, test_batch.isf_score_ids)

  # Print Test Summary
  print("Training ("+str(test_batch.docs.shape[0])+") Accuracy = {:.6f}".format(test_acc))
  # Estimate MRR on validation set
  mrr_score = mrr_metric(probs, test_batch.labels, test_batch.weights, test_batch.isf_score_ids)
  print("Training ("+str(test_batch.docs.shape[0])+") MRR = {:.6f}".format(mrr_score))
  # Estimate MAP score on validation set
  mapsc = map_score(probs, test_batch.labels, test_batch.weights, test_batch.isf_score_ids)
  print("Training ("+str(test_batch.docs.shape[0])+") MAP = {:.6f}".format(mapsc))

  # Writing test predictions and final summaries
  modelname = "step-a.model.ckpt.epoch-" + str(FLAGS.model_to_load)
  write_prediction_summaries(test_batch, probs, modelname, "training")



# ######################## Test Mode on Validation Data ###########################

def test_val(sess):
  """
  Test on validation Mode: Loads an existing model and test it on the validation set
  """
  if FLAGS.load_prediction != -1:
    print("====================================== [%d] ======================================" % (FLAGS.load_prediction))

  ### Prepare data for training
  print("Prepare vocab dict and read pretrained word embeddings ...")
  vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
  # vocab_dict contains _PAD and _UNK but not word_embedding_array

  print("Prepare test data ...")
  train_data = DataProcessor().prepare_news_data(vocab_dict, data_type="training")
  val_batch = batch_load_data(DataProcessor().prepare_news_data(vocab_dict,
                                                                data_type="validation",
                                                                normalizer=train_data.normalizer,
                                                                pca_model=train_data.pca_model))
  del train_data

  # Create Model with various operations
  model = MY_Model(sess, len(vocab_dict)-2)

  # # Initialize word embedding before training
  #print("Initialize word embedding vocabulary with pretrained embeddings ...")
  #sess.run(model.vocab_embed_variable.assign(word_embedding_array))

  # Select the model
  if (os.path.isfile(FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load))):
    selected_modelpath = FLAGS.train_dir+"/step-a.model.ckpt.epoch-"+str(FLAGS.model_to_load)
  else:
    print("Model not found in checkpoint folder.")
    exit(0)

  # Reload saved model and test
  print("Reading model parameters from %s" % selected_modelpath)
  model.saver.restore(sess, selected_modelpath)
  print("Model loaded.")

  # Initialize word embedding before training
  print("Initialize word embedding vocabulary with pretrained embeddings ...")
  sess.run(model.vocab_embed_variable.assign(word_embedding_array))

  # Test Accuracy and Prediction
  print("Performance on the validation data:")
  FLAGS.authorise_gold_label = False
  val_batch = batch_predict_with_a_model(val_batch,"validation", model, session=sess)
  FLAGS.authorise_gold_label = True
  probs = sess.run(model.probs,feed_dict={model.logits_placeholder: val_batch.logits})

  # Validation Accuracy and Prediction
  validation_acc = accuracy_qas_top(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids)

  # Print Validation Summary
  print("Validation (%s) Accuracy= %.6f" % (str(val_batch.docs.shape[0]),validation_acc))
  # Estimate MRR on validation set
  mrr_score = mrr_metric(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids)
  print("Validation (%s) MRR= %.6f" % (str(val_batch.docs.shape[0]),mrr_score))
  # Estimate MAP score on validation set
  mapsc = map_score(probs, val_batch.labels, val_batch.weights, val_batch.isf_score_ids)
  print("Validation (%s) MAP= %.6f" % (str(val_batch.docs.shape[0]),mapsc))

  if FLAGS.load_prediction != -1:
    fn = ''
    if FLAGS.filtered_setting:
      fn = "%s/step-a.model.ckpt.%s-top%d-isf-metrics" % (FLAGS.train_dir,"validation",FLAGS.topK)
    else:
      fn = "%s/step-a.model.ckpt.%s-metrics" % (FLAGS.train_dir,"validation")
    save_metrics(fn,FLAGS.load_prediction,validation_acc,mrr_score,mapsc)

  # Writing validation predictions, embeddings
  print("Writing final validation summaries and embeddings")
  modelname = "step-a.model.ckpt.epoch-" + str(FLAGS.model_to_load)
  write_prediction_summaries(val_batch, probs, modelname, "validation")
  write_cos_sim(val_batch.cos_sim, modelname + "-cos_sim","validation")


