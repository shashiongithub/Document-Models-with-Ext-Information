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
from tensorflow.python.ops import seq2seq
from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access


# from tf.nn import variable_scope
from my_flags import FLAGS

### Get Variable

def variable_on_cpu(name, shape, initializer, trainable=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    trainable: is trainable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def get_vocab_embed_variable(vocab_size):
  '''Returns vocab_embed_variable without any local initialization
  '''
  vocab_embed_variable = ""
  if FLAGS.trainable_wordembed:
    vocab_embed_variable = variable_on_cpu("vocab_embed", [vocab_size, FLAGS.wordembed_size], tf.constant_initializer(0), trainable=True)
  else:
    vocab_embed_variable = variable_on_cpu("vocab_embed", [vocab_size, FLAGS.wordembed_size], tf.constant_initializer(0), trainable=False)

  return vocab_embed_variable

def get_lstm_cell():
  """Define LSTM Cell
  """
  single_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.size) if (FLAGS.lstm_cell == "lstm") else tf.nn.rnn_cell.GRUCell(FLAGS.size)
  cell = single_cell
  if FLAGS.num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * FLAGS.num_layers)
  return cell

### Reshaping

def reshape_tensor2list(tensor, n_steps, n_input):
  """Reshape tensor [?, n_steps, n_input] to lists of n_steps items with [?, n_input]
  """
  # Prepare data shape to match `rnn` function requirements
  # Current data input shape (batch_size, n_steps, n_input)
  # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
  #
  # Permuting batch_size and n_steps
  tensor = tf.transpose(tensor, perm=[1, 0, 2], name='transpose')
  # Reshaping to (n_steps*batch_size, n_input)
  tensor = tf.reshape(tensor, [-1, n_input], name='reshape')
  # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
  tensor = tf.split(0, n_steps, tensor, name='split')
  return tensor

def reshape_list2tensor(listoftensors, n_steps, n_input):
  """Reshape lists of n_steps items with [?, n_input] to tensor [?, n_steps, n_input]
  """
  # Reverse of _reshape_tensor2list
  tensor = tf.concat(0, listoftensors, name="concat") # [n_steps * ?, n_input]
  tensor = tf.reshape(tensor, [n_steps, -1, n_input], name='reshape') # [n_steps, ?, n_input]
  tensor = tf.transpose(tensor, perm=[1, 0, 2], name="transpose") # [?, n_steps, n_input]
  return tensor

### Convolution, LSTM, RNNs

def multilayer_perceptron(final_output, weights, biases):
  """MLP over output with attention over enc outputs
  Args:
     final_output: [batch_size x 2*size]
  Returns:
     logit:  [batch_size x target_label_size]
  """

  # Layer 1
  layer_1 = tf.add(tf.matmul(final_output, weights["h1"]), biases["b1"])
  layer_1 = tf.nn.relu(layer_1)

  # Layer 2
  layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
  layer_2 = tf.nn.relu(layer_2)

  # output layer
  layer_out = tf.add(tf.matmul(layer_2, weights["out"]), biases["out"])

  return layer_out


def conv1d_layer_sentence_representation(sent_wordembeddings):
  """Apply mulitple conv1d filters to extract sentence respresentations
  Args:
  sent_wordembeddings: [None, max_sent_length, wordembed_size]
  Returns:
  sent_representations: [None, sentembed_size]
  """

  representation_from_filters = []

  output_channel = 0
  if FLAGS.handle_filter_output == "sum":
    output_channel = FLAGS.sentembed_size
  else: # concat
    fil_lens_to_test = (FLAGS.max_filter_length - FLAGS.min_filter_length + 1)
    output_channel = FLAGS.sentembed_size / fil_lens_to_test
    if (output_channel *  fil_lens_to_test != FLAGS.sentembed_size):
      print("Error: Make sure (output_channel *  FLAGS.max_filter_length) is equal to FLAGS.sentembed_size.")
      exit(0)

  for filterwidth in xrange(FLAGS.min_filter_length,FLAGS.max_filter_length+1):

    with tf.variable_scope("Conv1D_%d"%filterwidth) as scope:

      # Convolution
      conv_filter = variable_on_cpu("conv_filter_%d" % filterwidth, [filterwidth, FLAGS.wordembed_size, output_channel], tf.truncated_normal_initializer())
      # print(conv_filter.name, conv_filter.get_shape())
      conv = tf.nn.conv1d(sent_wordembeddings, conv_filter, 1, padding='VALID') # [None, out_width=(max_sent_length-(filterwidth-1)), output_channel]
      conv_biases = variable_on_cpu("conv_biases_%d" % filterwidth, [output_channel], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, conv_biases)
      conv = tf.nn.relu(pre_activation) #  [None, out_width, output_channel]
      # print(conv.name, conv.get_shape())

      # Max pool: Reshape conv to use max_pool
      conv_reshaped = tf.expand_dims(conv, 1) # [None, out_height:1, out_width, output_channel]
      # print(conv_reshaped.name, conv_reshaped.get_shape())
      out_height = conv_reshaped.get_shape()[1].value
      out_width = conv_reshaped.get_shape()[2].value
      # print(out_height,out_width)
      maxpool = tf.nn.max_pool(conv_reshaped, [1,out_height,out_width,1], [1,1,1,1], padding='VALID') # [None, 1, 1, output_channel]
      # print(maxpool.name, maxpool.get_shape())

      # Local Response Normalization
      maxpool_norm = tf.nn.lrn(maxpool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75) # Settings from cifar10
      # print(maxpool_norm.name, maxpool_norm.get_shape())

      # Get back to original dimension
      maxpool_sqz = tf.squeeze(maxpool_norm, [1,2]) # [None, output_channel]
      # print(maxpool_sqz.name, maxpool_sqz.get_shape())

    representation_from_filters.append(maxpool_sqz)
    # print(representation_from_filters)

  final_representation = []
  with tf.variable_scope("FinalOut") as scope:
    if FLAGS.handle_filter_output == "sum":
      final_representation = tf.add_n(representation_from_filters)
    else:
      final_representation = tf.concat(1, representation_from_filters)

  return final_representation

def simple_rnn(rnn_input, initial_state=None):
  """Implements Simple RNN
  Args:
  rnn_input: List of tensors of sizes [-1, sentembed_size]
  Returns:
  encoder_outputs, encoder_state
  """
  # Setup cell
  cell_enc = get_lstm_cell()

  # Apply dropout
  keep_prob = FLAGS.dropout if FLAGS.use_dropout else 1.0
  cell_enc = tf.nn.rnn_cell.DropoutWrapper(cell_enc,input_keep_prob=keep_prob)

  # Setup RNNs
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  rnn_outputs, rnn_state = tf.nn.rnn(cell_enc, rnn_input, dtype=dtype, initial_state=initial_state)
  # print(rnn_outputs)
  # print(rnn_state)

  return rnn_outputs, rnn_state

def simple_attentional_rnn(rnn_input, attention_state_list, initial_state=None):
  """Implements Simple RNN
  Args:
  rnn_input: List of tensors of sizes [-1, sentembed_size]
  attention_state_list: List of tensors of sizes [-1, sentembed_size]
  Returns:
  outputs, state
  """

  # Reshape attention_state_list to tensor
  attention_states = reshape_list2tensor(attention_state_list, len(attention_state_list), FLAGS.sentembed_size)

  # Setup cell
  cell = get_lstm_cell()

  # Apply dropout
  in_prob = FLAGS.dropout if FLAGS.use_dropout else 1.0
  out_prob = FLAGS.dropout if FLAGS.use_dropout_outatt else 1.0  
  cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=in_prob,output_keep_prob=out_prob)

  # Setup attentional RNNs
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

  # if initial_state == None:
  #   batch_size = tf.shape(rnn_input[0])[0]
  #   initial_state = cell.zero_state(batch_size, dtype)

  rnn_outputs, rnn_state = seq2seq.attention_decoder(rnn_input, initial_state, attention_states, cell,
                                                     output_size=None, num_heads=1, loop_function=None, dtype=dtype,
                                                     scope=None, initial_state_attention=False)
  # print(rnn_outputs)
  # print(rnn_state)
  return rnn_outputs, rnn_state



### Special decoders

def jporg_attentional_seqrnn_decoder(sents_ext, encoder_outputs, encoder_state, sents_labels, weights, biases):
  """
  Implements JP's special decoder: attention over encoder
  """

  # Setup cell
  cell_ext = get_lstm_cell()

  # Define Sequential Decoder
  with variable_scope.variable_scope("JP_Decoder"):
    state = encoder_state
    extractor_logits = []
    extractor_outputs = []
    prev = None
    for i, inp in enumerate(sents_ext):
      if prev is not None:
        with variable_scope.variable_scope("loop_function"):
          inp = _loop_function(inp, extractor_logits[-1], sents_labels[i-1])
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # Create Cell
      output, state = cell_ext(inp, state)
      prev = output

      # Convert output to logit
      with variable_scope.variable_scope("mlp"):
        combined_output = [] # batch_size, 2*size
        if FLAGS.doc_encoder_reverse:
          combined_output = tf.concat(1, [output, encoder_outputs[(FLAGS.max_doc_length - 1) - i]])
        else:
          combined_output = tf.concat(1, [output, encoder_outputs[i]])

        logit = multilayer_perceptron(combined_output, weights, biases)

      extractor_logits.append(logit)
      extractor_outputs.append(combined_output)

  return extractor_outputs, extractor_logits

### Private Functions

def _loop_function(current_inp, ext_logits, gold_logits):
  """ Update current input wrt previous logits
  Args:
  current_inp: [batch_size x sentence_embedding_size]
  ext_logits: [batch_size x target_label_size] [1, 0]
  gold_logits: [batch_size x target_label_size]
  Returns:
  updated_inp: [batch_size x sentence_embedding_size]
  """

  prev_logits = gold_logits
  if not FLAGS.authorise_gold_label:
    prev_logits = ext_logits
    prev_logits = tf.nn.softmax(prev_logits) # [batch_size x target_label_size]

  prev_logits = tf.split(1, FLAGS.target_label_size, prev_logits) # [[batch_size], [batch_size], ...]
  prev_weight_one = prev_logits[0]

  updated_inp = tf.mul(current_inp, prev_weight_one)
  # print(updated_inp)

  return updated_inp


### SoftMax and Predictions

def convert_logits_to_softmax(batch_logits, session=None,trans_np=True):
  """ Convert logits to probabilities
  batch_logits: [batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
  """
  # Convert logits [batch_size, FLAGS.max_doc_length, FLAGS.target_label_size] to probabilities
  batch_logits = tf.reshape(batch_logits, [-1, FLAGS.target_label_size])
  batch_softmax_logits = tf.nn.softmax(batch_logits)
  batch_softmax_logits = tf.reshape(batch_softmax_logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size])
  # Convert back to numpy array
  if trans_np:
    batch_softmax_logits = batch_softmax_logits.eval(session=session)
  return batch_softmax_logits

def convert_logits_to_sigmoid(batch_logits, session=None,trans_np=True):
  """ Applies sigmoid to logits (p(1) & p(0) are not mutually exclusive)
  batch_logits: [batch_size, FLAGS.max_doc_length, FLAGS.target_label_size]
  """
  # Convert logits [batch_size, FLAGS.max_doc_length, FLAGS.target_label_size] to probabilities
  batch_logits = tf.reshape(batch_logits, [-1, FLAGS.target_label_size])
  batch_softmax_logits = tf.nn.sigmoid(batch_logits)
  batch_softmax_logits = tf.reshape(batch_softmax_logits, [-1, FLAGS.max_doc_length, FLAGS.target_label_size])
  # Convert back to numpy array
  if trans_np:
    batch_softmax_logits = batch_softmax_logits.eval(session=session)
  return batch_softmax_logits
