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
from desanotools import *

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
		#return np.zeros(to_predict.shape)

class AutoEncoderTrainer(object):

	def __init__(self, Xgt):
		self.Xgt = Xgt
		self.autoencoder_model = None
		self.encoder_model = None

	def train(self):

		self.autoencoder_model, self.encoder_model = self.get_auto_encoder_model(self.Xgt.shape[1])

		self.autoencoder_model.fit(self.Xgt, self.Xgt, epochs=10, batch_size=256, validation_split=.1)

		Xgt_pred = self.autoencoder_model.predict(self.Xgt[0:100])
		Xgt_encoded = self.encoder_model.predict(self.Xgt[0:100])
		eval = self.autoencoder_model.evaluate(self.Xgt[0:100], self.Xgt[0:100])
		metrics_names = ["loss", "acc"]
		print("Train result : "+','.join([metrics_names[i]+" : "+str(eval[i]) for i in range(0, len(eval))]))

	def test(self):
		if self.encoder_model == None:
			print("Impossible to test encoder_model because it's not set")
			return

		Xgt_encoded = self.encoder_model.predict(self.Xgt[0:100])
		print(self.Xgt[0])
		print(Xgt_encoded[0])

		#encoder_model.save("encoder.h5")

		eds = euclidean_distances(Xgt_encoded, [Xgt_encoded[0]])
		#print(eds)

		for i in range(0, self.Xgt[0:100].shape[0]):
			print(self.Xgt[0:100][i].tolist(), eds[i][0])

	def save_model(self, model, file_path):
		model.save(file_path)

	def get_encoder_model(self, input_vec_length):
		model = Sequential()
		model.add(Dense(input_vec_length*2, input_shape=(input_vec_length,), activation = 'sigmoid'))
		model.add(Dense(input_vec_length, activation='relu'))

		model.compile(optimizer='adam', loss='mse', metrics=['acc'])
		print(model.summary())
		return model

	def get_auto_encoder_model(self, input_vec_length):
		input_vec = Input(shape=(input_vec_length,))
		encoded = Dense(units=3, activation='linear')(input_vec)
		decoded = Dense(units=3, activation='linear')(encoded)
		decoded = Dense(units=input_vec_length, activation='relu')(encoded)

		#best
		#input_vec = Input(shape=(input_vec_length,))
		#encoded = Dense(units=8, activation='relu')(input_vec)
		#decoded = Dense(units=input_vec_length, activation='relu')(encoded)

		autoencoder=Model(input_vec, decoded)
		autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

		encoder = Model(input_vec, encoded)
		return autoencoder, encoder

def load_autoencoder(autoencoder_filepath):
	#encoder_model = load_model("encoder.h5")
	return AutoEncoderLoader(autoencoder_filepath)
