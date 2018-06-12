####################################
# Author: Shashi Narayan
# Date: September 2016
# Project: Document Summarization
# H2020 Summa Project
# Comments: Jan 2017
# Improved for Reinforcement Learning
####################################

"""
Document Summarization System
"""
import tensorflow as tf
from my_flags import FLAGS
from train_test_utils import *

######################## Main Function ###########################

def main(_):
	meta_experiment_gpu_conf(FLAGS.exp_mode)


if __name__ == "__main__":
  tf.app.run()
