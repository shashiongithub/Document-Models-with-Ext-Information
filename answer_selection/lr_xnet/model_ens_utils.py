from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import math_ops
from sklearn.metrics import average_precision_score as aps
import pdb

from model_docsum import *
from model_utils import *


from my_flags import FLAGS

def sentence_extractor_nonseqrnn_titimgatt_orig(sents_ext, encoder_state, titleimages):
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
    logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='sidenet-logits')
  return rnn_extractor_output, logits


def policy_network_ens(vocab_embed_variable, document_placeholder, label_placeholder, isf_placeholder, idf_placeholder):
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
      
      """
      with variable_scope.variable_scope("Cosine_Similarity"):
        cos_similarity = calc_cos_similarity( document_sents_ext,
                                              document_sents_titimg[0]) # similarity with question
      """

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
              extractor_output, logits = sentence_extractor_nonseqrnn_titimgatt_orig(document_sents_ext, encoder_state, document_sents_titimg)

          elif (FLAGS.attend_encoder) and (len(document_sents_titimg) == 0):
              # JP model: attend encoder
              extractor_outputs, logits = sentence_extractor_seqrnn_docatt(document_sents_ext, encoder_outputs, encoder_state, label_placeholder)
          else:
              # Attend nothing
              extractor_output, logits = sentence_extractor_nonseqrnn_noatt(document_sents_ext, encoder_state)

      with tf.variable_scope('LogReg_Layer') as scope:
          probs = convert_logits_to_softmax(logits,None,False)
          one_probs = tf.reshape(tf.slice(probs,[0,0,0],[-1,-1,1]),[-1,FLAGS.max_doc_length]) # [bs,max_doc_len]
          # normalization, if requested
          if FLAGS.norm_feats:
              isf_placeholder = tf.nn.l2_normalize(isf_placeholder,dim=0)
              idf_placeholder = tf.nn.l2_normalize(idf_placeholder,dim=0)
              #sent_len_placeholder = tf.nn.l2_normalize(sent_len_placeholder,dim=0)
              #cnt_placeholder = tf.nn.l2_normalize(cnt_placeholder,dim=0)
          one_probs = tf.reshape(one_probs,[-1,1])
          #cos_similarity_flat = tf.reshape(cos_similarity,[-1,1])
          #sent_len_placeholder = tf.reshape(sent_len_placeholder,[-1,1])
          #cnt_placeholder = tf.reshape(cnt_placeholder,[-1,1])
          idf_placeholder = tf.reshape(idf_placeholder,[-1,1])
          isf_placeholder = tf.reshape(isf_placeholder,[-1,1])
          acum = [
              one_probs,
              isf_placeholder,
              idf_placeholder,
          ]
          nfeats = len(acum)
          acum = tf.concat(1,acum) # [batch_size * max_doc_len, nfeats]
          h_ens = tf.reshape(acum,[-1,FLAGS.max_doc_length,nfeats])

          # Define Variables
          weight = variable_on_cpu('weight', [nfeats, FLAGS.target_label_size], tf.random_normal_initializer())
          bias = variable_on_cpu('bias', [FLAGS.target_label_size], tf.random_normal_initializer())
          ensemble_output_forlogits = acum
          logits = tf.matmul(ensemble_output_forlogits, weight) + bias
          logits = tf.reshape(logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size], name='final-logits')
  return h_ens, logits


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
    logits: logits from LR layer. [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    labels: Sentence extraction gold levels [FLAGS.batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
    weights: Weights to avoid padded part [FLAGS.batch_size, FLAGS.max_doc_length]
    Returns:
    Cross-entropy Cost
    """
    with tf.variable_scope('CrossEntropyLoss') as scope:
        # Reshape logits and labels to match the requirement of softmax_cross_entropy_with_logits
        logits = tf.reshape(logits, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]
        labels = tf.reshape(labels, [-1, FLAGS.target_label_size]) # [FLAGS.batch_size*FLAGS.max_doc_length, FLAGS.target_label_size]

        cross_entropy = ''
        if FLAGS.lr_activation == 'sigmoid':
          cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels) # [FLAGS.batch_size*FLAGS.max_doc_length,FLAGS.target_label_size] <<<--- omfg
          cross_entropy = tf.reduce_sum(cross_entropy,reduction_indices=1) # [FLAGS.batch_size*FLAGS.max_doc_length]
        else:
          cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels) # [FLAGS.batch_size*FLAGS.max_doc_length]
        cross_entropy = tf.reshape(cross_entropy, [-1, FLAGS.max_doc_length])  # [FLAGS.batch_size, FLAGS.max_doc_length]

        if FLAGS.weighted_loss:
            cross_entropy = tf.mul(cross_entropy, weights)

        # Cross entroy / document
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) # [FLAGS.batch_size]
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='crossentropy')

        tf.add_to_collection('cross_entropy_loss', cross_entropy_mean)

    return cross_entropy_mean
