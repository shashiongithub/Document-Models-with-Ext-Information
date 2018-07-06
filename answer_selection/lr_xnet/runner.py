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

import sys
sys.path.append('../../common')

import tensorflow as tf
from my_flags import FLAGS
from train_test_utils import meta_experiment_gpu_conf

######################## Main Function ###########################

def main(_):
	meta_experiment_gpu_conf(FLAGS.exp_mode)


if __name__ == "__main__":
  tf.app.run()
