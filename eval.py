#! /usr/bin/env python
# encoding:utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import new_data_helper
import word2vec_helpers
from text_cnn import TextCNN
import csv
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("predict_file_path", "./data/predict_first.csv", "text data source to evaluate.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# validate
# ==================================================

# validate checkout point file
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# validate word2vec model file
trained_word2vec_model_file = os.path.join(FLAGS.checkpoint_dir, "..", "trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))

# validate training params file
training_params_file = os.path.join(FLAGS.checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

# Load params
params = new_data_helper.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_document_length = int(params['max_document_length'])

# Load data
if FLAGS.eval_train:
    x_raw, ids = new_data_helper.load_predict_text(FLAGS.predict_file_path)
    print(len(ids))
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0, 0, 0, 0]

# Get Embedding vector x_test
sentences, max_document_length = new_data_helper.padding_sentences(x_raw, '.', padding_sentence_len=max_document_length)
# print(sentences)
# 如果测试的文件过大，则容易出现，x_test.shape出错的问题，因此将测试集分割测试
# Collect the predictions here
all_predictions = []
print(len(sentences))
for i in range(len(sentences) / 100):
    print(i)
    print(len(sentences) / 100)
    print(sentences[i * 100 : (i + 1) * 100])
    x_test = np.array(word2vec_helpers.embedding_sentences(sentences[i * 100 : (i + 1) * 100], file_to_load=trained_word2vec_model_file))
    #print(x_test)
    print("x_test.shape = {}".format(x_test.shape))

    # Evaluation
    # ==================================================
    print("\nEvaluating...\n")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
    
            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
    
            # Generate batches for one epoch
            batches = new_data_helper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
    
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                print(all_predictions)
                print(len(all_predictions))
               
                    # Save the evaluation to a csv
predictions_human_readable = np.column_stack((ids, all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)


