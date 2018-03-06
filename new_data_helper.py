#encoding:utf-8

import numpy as np 
import re
import itertools
from collections import Counter
import os
import word2vec_helpers
import time
import pickle
import pandas as pd 

def load_predict_text(predict_file_path):
	predict_data = pd.read_csv(predict_file_path,header = 0, encoding='utf8')
	lines = []
	for line in predict_data['Discuss']:
		lines.append(seperate_line(line))
	return [lines,predict_data['Id']]

#加载csv文件中的文本和标签
def load_train_text_and_tag(train_data_path):
	train_data = pd.read_csv(train_data_path,header = 0, encoding='utf8')
	#每一行的训练文本去标点
	lines = []
	for line in train_data['Discuss']:
		lines.append(seperate_line(line))
	tags = []
	tag_all = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
	for item in train_data['Score']:
		tags.append(tag_all[item-1])
        print(lines[0])

	return [lines,np.array(tags)]

#将句子补偿为最大长度
def padding_sentences(input_sentences,padding_token,padding_sentence_len=None):
	sentences = [sentence.split(' ') for sentence in input_sentences]
	max_sentence_len = padding_sentence_len if padding_sentence_len is not None else max([len(sentence) for sentence in sentences])
	for sentence in sentences:
		if len(sentence) > max_sentence_len:
			sentence = sentence[:max_sentence_len]
		else:
			sentence.extend([padding_token]*(max_sentence_len-len(sentence)))
	return (sentences,max_sentence_len)

#产生每次训练的输入
def batch_iter(data,batch_size,num_epochs,shuffle=True):
	data = np.array(data)
	data_size = len(data)
	num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
	for epoch in range(num_epochs):

		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batchs_per_epoch):
			start_idx = batch_num * batch_size
			end_idx = min((batch_num + 1) * batch_size,data_size)
			yield shuffled_data[start_idx : end_idx]

def mkdir_if_not_exits(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

#将连续的句子按字隔开
def seperate_line(line):
	line = clean_str(line)
	return ''.join([word + ' ' for word in line])

#去掉句子中的标点符号
def clean_str(string):
	string = re.sub(ur"[^\u4e00-\u9fff]"," ",string)
	string = re.sub(r"\s{2,}"," ",string)
	return string.strip()

#保存数据字典
def saveDict(input_dict,output_file):
	with open(output_file,'wb') as f:
		pickle.dump(input_dict,f)

def loadDict(dict_file):
	output_dict = None
	with open(dict_file,'rb') as f:
		output_dict = pickle.load(f)
	return output_dict
