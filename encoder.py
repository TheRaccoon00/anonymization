"""
File: encoder.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database anonymizer for DARC competition
"""

import pandas as pd
import time, random, sys, os, argparse
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from threading import Thread
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

#turnn off fucking ugly warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from attack_threaded import *


from keras_preprocessing.text import Tokenizer, tokenizer_from_json
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import sequence
from keras.layers.core import Layer, Dense, Activation, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers import Input, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPool2D, Conv1D, Bidirectional, GRU, concatenate, merge
from keras.models import Sequential, model_from_json, load_model
from keras import optimizers, Model
from keras import backend as K
from keras import regularizers

#change this values in function of what is in anonymized data
use_index = False			#default False
use_dates = True			#default True
use_hours = False			#default False
use_items = True			#default True
use_scaler = False			#default False

#change this values to speed up research by reducing space research
force_year_equality = True #default False
force_month_equality = True#default False
force_day_equality = False	#default False
force_item_equality = True	#default False
force_qtt_equality = True	#default False

nb_threads = 1				#increase this value to speed up compute

show_result = False

#do not change this values
id_index = 0
date_index = 1
hours_index = 2
item_index = 3
price_index = 4
qtt_index = 5

gt_path = "ground_truth.csv"		#ground truth file path
dt_path = ""		#anonymized file path
out_path = ""		#output file path


class AutoEncoderLoader(object):
	def __init__(self, file_path):
		self.file_path = file_path
		#K.get_session().run(tf.global_variables_initializer())
		self.sess = K.get_session()

		self.model = load_model(self.file_path)
		#self.model._make_predict_function()

		self.graph = tf.get_default_graph()

	def predict(self, to_predict):
		#with tf.Graph().as_default() as graph:
		with self.graph.as_default():
			set_session(self.sess)
			#K.get_session().run(tf.local_variables_initializer())
			predict = self.model.predict(to_predict)
		return predict

def main():
	global id_index, date_index, item_index, hours_index, price_index, qtt_index, use_scaler

	############################################################################
	#read and convert datasets
	#gt is the ground_truth dataset and dt is the anonymized dataset to crack (dt for data)
	print("Reading datasets...")

	#works for S_Godille_Table_1, S_Godille_Table_2, S_Godille_Table_3
	gt = pd.read_csv(gt_path, sep=",")

	print("Converting datasets...")
	gtn = np.asarray(gt)
	#[17850 '2010/12/01' '08:26' '85123A' 2.55 6]

	############################################################################
	#delete axis in function of what we want to do : check data_index, item_index, hours_index
	print("Deleting useless columns...")
	if not use_index:
		gtn = np.delete(gtn, id_index, axis=1)
		id_index = id_index - 1
		date_index = date_index - 1
		hours_index = hours_index - 1
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	if not use_dates:
		gtn = np.delete(gtn, date_index, axis=1)
		date_index = date_index - 1
		hours_index = hours_index - 1
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	if not use_hours:
		gtn = np.delete(gtn, hours_index, axis=1)
		hours_index = hours_index - 1
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	if not use_items:
		gtn = np.delete(gtn, item_index, axis=1)
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	############################################################################
	#if we use date_index, so we convert XX/XX/XXXX to XX, XX, XX and ad it to final vector
	if use_dates:
		print("Splitting dates...")
		gtn = split_date(gtn, date_index)
		item_index = item_index + 2 #deleted 1 item, added 3 so add 2 to item_index

	############################################################################
	#vectorization of data

	#mega scaler of the death
	#sscaler = StandardScaler()
	sscaler = MaxAbsScaler()

	#############################
	#step 1 : learn how to vectorize from ground_truth
	#learnt vectorization is stored in dict and dict_rev for the reversible transformation
	#learn how to convert str items to int/float items and store it in tables for date/hour/item
	print("Learning transformation from ground_truth...")
	print("gtn", gtn[0])
	print(id_index, date_index, hours_index, item_index, price_index, qtt_index)
	#trans_table_date_y, trans_table_date_y_rev,\
	#trans_table_date_m,trans_table_date_m_rev,\
	#trans_table_date_d, trans_table_date_d_rev,\
	#trans_table_hours, trans_table_hours_rev,\
	#trans_table_item, trans_table_item_rev,\
	#Xgt = learn_panda_to_vector(gtn.copy(), sscaler)

	X = gtn.copy()

	trans_table_date_y = dict()
	trans_table_date_y_rev = dict()
	trans_table_date_m = dict()
	trans_table_date_m_rev = dict()
	trans_table_date_d = dict()
	trans_table_date_d_rev = dict()
	trans_table_hours = dict()
	trans_table_hours_rev = dict()
	trans_table_item = dict()
	trans_table_item_rev = dict()

	#apply vectorization
	if use_dates:
		print("Transforming date")
		trans_table_date_y, trans_table_date_y_rev, X = vectorize(X, date_index)
		trans_table_date_m, trans_table_date_m_rev, X = vectorize(X, date_index+1)
		trans_table_date_d, trans_table_date_d_rev, X = vectorize(X, date_index+2)


	if use_hours:
		print("Transforming hours")
		trans_table_hours, trans_table_hours_rev, X = vectorize(X, hours_index)

	if use_items:
		print("Transforming id_item")
		trans_table_item, trans_table_item_rev, X = vectorize(X, item_index)

	#print(X[0]) => [17850 0 10 241 2.55 6]

	#apply scaler to input
	if use_scaler:
		X = scaler.fit_transform(X)
#	return 	trans_table_date_y, trans_table_date_y_rev,\
#			trans_table_date_m, trans_table_date_m_rev,\
#			trans_table_date_d, trans_table_date_d_rev,\
#			trans_table_hours, trans_table_hours_rev, trans_table_item, trans_table_item_rev, X

	print(trans_table_date_y)
	Xgt = X.astype(np.float64)
	print(Xgt[0])

	train(Xgt)
	test(Xgt)

def get_encoder_model(input_vec_length):
	model = Sequential()
	model.add(Dense(input_vec_length*2, input_shape=(input_vec_length,), activation = 'sigmoid'))
	model.add(Dense(input_vec_length, activation='relu'))

	model.compile(optimizer='adam', loss='mse', metrics=['acc'])
	print(model.summary())
	return model

def get_auto_encoder_model(input_vec_length):
	input_vec = Input(shape=(input_vec_length,))
	encoded = Dense(units=8, activation='relu')(input_vec)
	encoded = Dense(units=3, activation='relu')(encoded)
	decoded = Dense(units=64, activation='relu')(encoded)
	decoded = Dense(units=input_vec_length, activation='relu')(decoded)

	autoencoder=Model(input_vec, decoded)
	autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

	encoder = Model(input_vec, encoded)
	return autoencoder, encoder

def train(Xgt):

	autoencoder_model, encoder_model = get_auto_encoder_model(Xgt.shape[1])

	autoencoder_model.fit(Xgt,Xgt,epochs=10,batch_size=256, validation_split=.1)

	Xgt_pred = autoencoder_model.predict(Xgt[0:100])
	Xgt_encoded = encoder_model.predict(Xgt[0:100])
	print(Xgt[0])
	print(Xgt_encoded[0])
	print(Xgt_pred[0])
	eval = autoencoder_model.evaluate(Xgt[0:100], Xgt[0:100])
	print(autoencoder_model.metrics_names, eval)

def test(Xgt):
	encoder_model = load_model("encoder.h5")

	Xgt_encoded = encoder_model.predict(Xgt[0:100])
	print(Xgt[0])
	print(Xgt_encoded[0])

	encoder_model.save("encoder.h5")

	eds = euclidean_distances(Xgt_encoded, [Xgt_encoded[0]])
	#print(eds)

	for i in range(0, Xgt[0:100].shape[0]):
		print(Xgt[0:100][i].tolist(), eds[i][0])

def load_autoencoder(autoencoder_filepath):
	#encoder_model = load_model("encoder.h5")
	return AutoEncoderLoader(autoencoder_filepath)



if __name__ == "__main__":
	main()
