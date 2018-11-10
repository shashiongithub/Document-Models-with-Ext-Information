####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project

# v1.2 XNET-QA
#   author: Ronald Cardenas
####################################

"""
Document Summarization/Question Answering Modules and Models
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
        logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='final-logits')
    return rnn_extractor_output, logits

def sentence_extractor_nonseqrnn_titimgatt(sents_ext, encoder_state, titleimages):
    """Implements Sentence Extractor: Non-sequential RNN with attention over title-images
    Args:
    sents_ext: Embedding of sentences to label for extraction
    encoder_state: encoder_state
    titleimages: Embeddings of title and images in the document
    Returns:
    extractor output and logits
    """

    # Define Variables
    weight = variable_on_cpu('weight', [FLAGS.size, FLAGS.target_label_size], tf.random_normal_initializer())
    bias = variable_on_cpu('bias', [FLAGS.target_label_size], tf.random_normal_initializer())

    # Get RNN output
    rnn_extractor_output, _ = simple_attentional_rnn(sents_ext, titleimages, initial_state=encoder_state)

    with variable_scope.variable_scope("Reshape-Out"):
      rnn_extractor_output = reshape_list2tensor(rnn_extractor_output, FLAGS.max_doc_length, FLAGS.size)

      # Get Final logits without softmax
      extractor_output_forlogits = tf.reshape(rnn_extractor_output, [-1, FLAGS.size])
      logits = tf.matmul(extractor_output_forlogits, weight) + bias
      # logits: [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
      logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='final-logits')
    return rnn_extractor_output, logits


def sentence_extractor_nonseqrnn_qa(sents_ext, encoder_state, titleimages, isf_scores, idf_scores, locisf_scores):
    """Implements Sentence Extractor: Non-sequential RNN with attention over title-images
    Args:
    sents_ext: Embedding of sentences to label for extraction
    encoder_state: encoder_state
    titleimages: Embeddings of title and images in the document
    isf_scores: [ [batch_size x 1]..]max_doc_length
    Returns:
    extractor output and logits
    """

    # Define Variables
    weight = variable_on_cpu('weight', [FLAGS.size, FLAGS.target_label_size], tf.random_normal_initializer(seed=seed))
    bias = variable_on_cpu('bias', [FLAGS.target_label_size], tf.random_normal_initializer(seed=seed))

    # Get RNN output
    rnn_extractor_output, _ = attentional_isf_rnn(sents_ext, titleimages, isf_scores, idf_scores, locisf_scores, initial_state=encoder_state)

    with variable_scope.variable_scope("Reshape-Out"):
      rnn_extractor_output = reshape_list2tensor(rnn_extractor_output, FLAGS.max_doc_length, FLAGS.size)

      # Get Final logits without softmax
      extractor_output_forlogits = tf.reshape(rnn_extractor_output, [-1, FLAGS.size])
      logits = tf.matmul(extractor_output_forlogits, weight) + bias
      # logits: [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
      logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='final-logits')
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


def resize_feat_to_emb(feat):
    """
    [batch_size,max_doc_len] -> [batch_size,max_doc_len,sentemb_size]
    """
    flat_feat = tf.reshape(feat,[-1,1])
    temp = tf.concat(1,[flat_feat] * FLAGS.sentembed_size)
    sph_rs = tf.reshape(temp,shape=[-1,FLAGS.max_doc_length,FLAGS.sentembed_size])
    scores_list = reshape_tensor2list(sph_rs, FLAGS.max_doc_length, FLAGS.sentembed_size)
    return scores_list



def policy_network(vocab_embed_variable, document_placeholder, label_placeholder, get_cos=False):
    """Build the policy core network.
    Args:
    vocab_embed_variable: [vocab_size, FLAGS.wordembed_size], embeddings without PAD and UNK
    document_placeholder: [None,(FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                 FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.max_sent_length]
    label_placeholder: Gold label [None, FLAGS.max_doc_length, FLAGS.target_label_size], only used during cross entropy training of JP's model.
    Returns:
    Outputs of sentence extractor and logits without softmax
    """

    with tf.variable_scope('PolicyNetwork') as scope:

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
            document_word_embedding = tf.reshape(document_word_embedding, [-1, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length),
                                                                           FLAGS.max_sent_length, FLAGS.wordembed_size])
            # print(document_word_embedding)

        ### Convolution Layer
        with tf.variable_scope('ConvLayer') as scope:
            document_word_embedding = tf.reshape(document_word_embedding, [-1, FLAGS.max_sent_length, FLAGS.wordembed_size])
            document_sent_embedding = conv1d_layer_sentence_representation(document_word_embedding) # [None, sentembed_size]
            document_sent_embedding = tf.reshape(document_sent_embedding, [-1, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.sentembed_size])
            # print(document_sent_embedding)

        ### Reshape Tensor to List [-1, (max_doc_length+max_title_length+max_image_length), sentembed_size] -> List of [-1, sentembed_size]
        with variable_scope.variable_scope("ReshapeDoc_TensorToList"):
            document_sent_embedding = reshape_tensor2list(document_sent_embedding, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                    FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.sentembed_size)

        # document_sents_enc
        document_sents_enc = document_sent_embedding[:FLAGS.max_doc_length]
        if FLAGS.doc_encoder_reverse:
            document_sents_enc = document_sents_enc[::-1]

        # document_sents_ext
        document_sents_ext = document_sent_embedding[:FLAGS.max_doc_length]

        # document_sents_titimg
        document_sents_titimg = document_sent_embedding[FLAGS.max_doc_length:]

        with variable_scope.variable_scope("Cosine_Similarity"):
            cos_similarity = [] if not get_cos else \
                calc_cos_similarity(document_sents_ext,document_sents_titimg[0]) # similarity with question

        ### Document Encoder
        with tf.variable_scope('DocEnc') as scope:
            encoder_outputs, encoder_state = simple_rnn(document_sents_enc)

        ### Sentence Label Extractor
        with tf.variable_scope('SentExt') as scope:
            if (FLAGS.attend_encoder) and (len(document_sents_titimg) != 0):
                # Multiple decoder
                print("Multiple decoder is not implement yet.")
                exit(0)
                # # Decoder to attend captions
                # attendtitimg_extractor_output, _ = simple_attentional_rnn(document_sents_ext, document_sents_titimg, initial_state=encoder_state)
                # # Attend previous decoder
                # logits = sentence_extractor_seqrnn_docatt(document_sents_ext, attendtitimg_extractor_output, encoder_state, label_placeholder)

            elif (not FLAGS.attend_encoder) and (len(document_sents_titimg) != 0):
                # Attend only titimages during decoding
                extractor_output, logits = sentence_extractor_nonseqrnn_titimgatt(document_sents_ext, encoder_state, document_sents_titimg)

            elif (FLAGS.attend_encoder) and (len(document_sents_titimg) == 0):
                # JP model: attend encoder
                extractor_outputs, logits = sentence_extractor_seqrnn_docatt(document_sents_ext, encoder_outputs, encoder_state, label_placeholder)

            else:
                # Attend nothing
                extractor_output, logits = sentence_extractor_nonseqrnn_noatt(document_sents_ext, encoder_state)

    # print(extractor_output)
    # print(logits)
    return extractor_output, logits, cos_similarity


def policy_network_xnet_plus_qa(vocab_embed_variable, document_placeholder, label_placeholder,
    isf_scores_placeholder, idf_scores_placeholder, locisf_scores_placeholder):
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

    with tf.variable_scope('PolicyNetwork') as scope:
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
            document_word_embedding = tf.reshape(document_word_embedding, [-1, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length),
                                                                           FLAGS.max_sent_length, FLAGS.wordembed_size])
        ### Convolution Layer
        with tf.variable_scope('ConvLayer') as scope:
            document_word_embedding = tf.reshape(document_word_embedding, [-1, FLAGS.max_sent_length, FLAGS.wordembed_size])
            document_sent_embedding = conv1d_layer_sentence_representation(document_word_embedding) # [None, sentembed_size]
            document_sent_embedding = tf.reshape(document_sent_embedding, [-1, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.sentembed_size])
        ### Reshape Tensor to List [-1, (max_doc_length+max_title_length+max_image_length), sentembed_size] -> List of [-1, sentembed_size]
        with variable_scope.variable_scope("ReshapeDoc_TensorToList"):
            document_sent_embedding = reshape_tensor2list(document_sent_embedding, (FLAGS.max_doc_length + FLAGS.max_title_length + FLAGS.max_image_length +
                                                                                    FLAGS.max_firstsentences_length + FLAGS.max_randomsentences_length), FLAGS.sentembed_size)
            isf_scores_list = []
            idf_scores_list = []
            locisf_scores_list = []
            if FLAGS.use_locisf:
              locisf_scores_list = resize_feat_to_emb(locisf_scores_placeholder)
            if FLAGS.use_isf:
              isf_scores_list = resize_feat_to_emb(isf_scores_placeholder)
            if FLAGS.use_idf:
              idf_scores_list = resize_feat_to_emb(idf_scores_placeholder)
              
        # document_sents_enc
        document_sents_enc = document_sent_embedding[:FLAGS.max_doc_length]
        if FLAGS.doc_encoder_reverse:
            document_sents_enc = document_sents_enc[::-1]

        # document_sents_ext
        document_sents_ext = document_sent_embedding[:FLAGS.max_doc_length]

        # document_sents_titimg
        document_sents_titimg = document_sent_embedding[FLAGS.max_doc_length:]

        with variable_scope.variable_scope("Cosine_Similarity"):
            cos_similarity = calc_cos_similarity(document_sents_ext,document_sents_titimg[0]) # similarity with question
            sent_emb = reshape_list2tensor(document_sents_ext,FLAGS.max_doc_length,FLAGS.sentembed_size)
            ques_emb = reshape_list2tensor(document_sents_titimg,1,FLAGS.sentembed_size)

        ### Document Encoder
        with tf.variable_scope('DocEnc') as scope:
            encoder_outputs, encoder_state = simple_rnn(document_sents_enc)

        ### Sentence Label Extractor
        with tf.variable_scope('SentExt') as scope:
            if (FLAGS.attend_encoder) and (len(document_sents_titimg) != 0):
                # Multiple decoder
                print("Multiple decoder is not implement yet.")
                exit(0)

            elif (not FLAGS.attend_encoder) and (len(document_sents_titimg) != 0):
                # Attend only titimages during decoding
                extractor_output, logits = sentence_extractor_nonseqrnn_qa(
                                                document_sents_ext, encoder_state, document_sents_titimg,
                                                isf_scores_list, idf_scores_list,locisf_scores_list)

            elif (FLAGS.attend_encoder) and (len(document_sents_titimg) == 0):
                # JP model: attend encoder
                extractor_outputs, logits = sentence_extractor_seqrnn_docatt(document_sents_ext, encoder_outputs, encoder_state, label_placeholder)

            else:
                # Attend nothing
                extractor_output, logits = sentence_extractor_nonseqrnn_noatt(document_sents_ext, encoder_state)

    return extractor_output, logits, cos_similarity,sent_emb,ques_emb


def policy_network_paircnn_qa(vocab_embed_variable, document_placeholder):
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

    with tf.variable_scope('PolicyNetwork') as scope:

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



def calc_cos_similarity(sentences_emb,question_emb):
  '''
  Calculates cosine similarity between the article sentences and question
  Args:
    sentences_emb: [ batch_size,sentembed_size]. max_doc_length
    question_emb:  batch_size,sentences_emb
  Returns:
    cosine similarity: [batch_size, max_doc_length]
  '''
  sims = []
  q_mod = tf.sqrt(tf.reduce_sum(tf.mul(question_emb,question_emb),1))
  for i,si in enumerate(sentences_emb):
    si_mod = tf.sqrt(tf.reduce_sum(tf.mul(si,si),1))
    dot_prod = tf.reduce_sum(tf.mul(si,question_emb),1)
    csim = tf.div(dot_prod,tf.mul(si_mod,q_mod))
    csim = tf.reshape(csim,[-1,1])
    sims.append(csim)
  sims = tf.concat(1,sims)
  sims = tf.select(tf.is_nan(sims),tf.zeros_like(sims),sims) # case whe sent_emb is zero (padded sentence)
  return sims


def cross_entropy_loss(logits, labels, weights):
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
        logits = tf.reshape(logits, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
        labels = tf.reshape(labels, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels) # [FLAGS.batch_size*FLAGS.max_doc_length]
        cross_entropy = tf.reshape(cross_entropy, [-1, FLAGS.max_doc_length])  # [FLAGS.batch_size, FLAGS.max_doc_length]
        if FLAGS.weighted_loss:
            cross_entropy = tf.mul(cross_entropy, weights)

        # Cross entroy / document
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) # [FLAGS.batch_size]
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='crossentropy')

        tf.add_to_collection('cross_entropy_loss', cross_entropy_mean)
        # # # The total loss is defined as the cross entropy loss plus all of
        # # # the weight decay terms (L2 loss).
        # # return tf.add_n(tf.get_collection('losses'), name='total_loss')
    return cross_entropy_mean


def cross_entropy_loss_paircnn_qa(logits, labels):
    """Estimate cost of predictions
    Add summary for "cost" and "cost/avg".
    Args:
    logits: Logits from inference(). [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    Returns:
    Cross-entropy Cost
    """
    with tf.variable_scope('CrossEntropyLoss') as scope:
        # Reshape logits and labels to match the requirement of softmax_cross_entropy_with_logits
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels) # [FLAGS.batch_size]
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
  accuracy = correct / total
  return accuracy



def mrr_metric(probs,labels,weights,scores,data_type):
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
    labels = labels * weights
  mask = labels.sum(axis=1) > 0
  mrr = 0.0
  total = 0.0
  for i in range(bs):
    if mask[i]==0:
      continue
    srt_ref = []
    # tie breaking: earliest in list
    if FLAGS.tie_break=="first":
        srt_ref = [(-x,j) for j,x in enumerate(one_prob[i,:])]
        srt_ref.sort()
    # tie breaking: last in list
    else:
        srt_ref = [(x,j) for j,x in enumerate(one_prob[i,:])]
        srt_ref.sort(reverse=True)

    rel_rank = 0.0
    for idx_retr,(x,j) in enumerate(srt_ref):
        if labels[i,j]==1:
            rel_rank = 1.0 + idx_retr
            break

    mrr += 1.0/rel_rank # accumulate inverse rank
    total += 1.0
  mrr /= total
  return mrr



def map_score(probs,labels,weights,scores,data_type):
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
  mask = labels.sum(axis=1) > 0
  _map = 0.0
  total = 0.0
  for i in range(bs):
    if mask[i]==0:
      continue
    srt_ref = []
    # tie breaking: earliest in list
    if FLAGS.tie_break=="first":
        srt_ref = [(-x,j) for j,x in enumerate(one_prob[i,:])]
        srt_ref.sort()
    # tie breaking: last in list
    else:
        srt_ref = [(x,j) for j,x in enumerate(one_prob[i,:])]
        srt_ref.sort(reverse=True)
    aps = 0.0
    n_corr = 0.0
    for idx_retr,(x,j) in enumerate(srt_ref):
        if labels[i,j]==1:
            n_corr += 1.0
            aps += (n_corr / (idx_retr+1))
            # break
    aps /= labels[i,:].sum()
    _map += aps
    total += 1.0
  _map /= total
  return _map



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


