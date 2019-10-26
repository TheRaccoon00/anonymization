"""
File: attack.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database anonymizer for DARC competition
"""

import pandas as pd
import time, random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

#turnn off fucking ugly warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def vectorize(X, num_col):
	"""
	Create a table wich translate an item to a unique id
	Return the translate table, inverse of the translate table and translated data
	Example :
	Input :
	=> X = [["AA", "AB", ... ], [ ...],  [ ...]]; num_col = 0;

	Output :
	=> trans_table = { "AA": 0, "BB": 1, ...}
	=> trans_table_rev = { 0: "AA", 1: "AB", ...}
	=> X = [[0, 1, ... ], [ ...],  [ ...]]
	"""
	item_ids_set = set(X[:,num_col])
	trans_table = dict()
	for new_id, id in enumerate(item_ids_set):
		trans_table[id] = new_id
	for i in range(0, X[:,num_col].shape[0]):
		#print(df["id_item"][i])
		X[i][num_col]=trans_table[X[i][num_col]]

	trans_table_rev = {v: k for k, v in trans_table.items()}
	return trans_table, trans_table_rev, X

def learn_panda_to_vector(df, scaler):
	X = np.asarray(df)

	#apply vectorization
	print("Transforming date")
	trans_table_date, trans_table_date_rev, X = vectorize(X, 1)

	print("Transforming hours")
	trans_table_hours, trans_table_hours_rev, X = vectorize(X, 2)

	print("Transforming id_item")
	trans_table_item, trans_table_item_rev, X = vectorize(X, 3)

	#print(X[0]) => [17850 0 10 241 2.55 6]

	#apply scaler to input
	X = scaler.fit_transform(X)
	return trans_table_date, trans_table_date_rev, trans_table_hours, trans_table_hours_rev, trans_table_item, trans_table_item_rev, X

def apply_transform(dfn, trans_table, num_col):
	"""
	apply learnt vectorization and return trans_table (and reversed version)
	in case of we had to add new entries to trans_table
	"""
	for i in range(0, dfn[:,num_col].shape[0]):
		#if key is not in trans_table
		if dfn[i][num_col] not in trans_table.keys():
			#we add new entry to the table
			new_entries_values = list(set(list(range(0, 2*len(trans_table.keys()))))-set(trans_table.keys()))
			new_entry_value = random.choice(new_entries_values)
			trans_table[dfn[i][num_col]] = new_entry_value
			#print("adding : ", dfn[i][num_col], new_entry_value)

		#apply transform
		dfn[i][num_col]=trans_table[dfn[i][num_col]]

	#get reversible table
	trans_table_rev = {v: k for k, v in trans_table.items()}
	return dfn, trans_table, trans_table_rev

def get_similar(Xgt, Xdt_row, return_length=10):
	"""
	return closest rows of dtn from gtn
	"""
	#print(cosine_similarity([Xgt[0]], [Xdt_row])[0][0]) #return the similarity of Xgt[0] and Xdt_row => 0.5008987774997233
	#print(cosine_similarity([Xgt[0]], [Xgt[0]])[0][0]) #return the similarity of Xgt[0] and Xgt[0] => 1.0
	#get distance to origin euclidean_distances([[0, 1], [1, 1]], [[0, 0]]) => array([[1.], [1.41421356]])

	#similarities = sorted(enumerate([cosine_similarity([Xgt_row], [Xdt_row])[0][0] for Xgt_row in Xgt]), key = lambda x: int(x[1]))
	eds = euclidean_distances(Xgt, [Xdt_row])
	#decrease dimension
	eds = [ed[0] for ed in eds]

	#list of distances from ground_truth vectors to Xdt_row sorted from lower to higher with corresponding index in Xgt
	distances = sorted(enumerate(eds), key = lambda x: int(x[1]))
	#print("max distance : ", max([d[1] for d in distances]))
	#print("min distance : ", min([d[1] for d in distances]))

	similar_rows = []
	similar_rows_score = []
	for i in range(0, return_length):
		similar_rows.append(Xgt[distances[i][0]])
		similar_rows_score.append(distances[i][1])

	return np.asarray(similar_rows), similar_rows_score

def reverse_vector(vec, scaler, trans_table_date_rev, trans_table_hours_rev, trans_table_item_rev):
	unscaled_vec = list(scaler.inverse_transform([vec])[0])
	unscaled_vec[1] = trans_table_date_rev[int(unscaled_vec[1])]
	unscaled_vec[2] = trans_table_hours_rev[int(unscaled_vec[2])]
	unscaled_vec[3] = trans_table_item_rev[int(unscaled_vec[3])]
	return unscaled_vec

def main():
	print("Reading datasets...")
	#load gt and dt is the anonymized dataset
	gt = pd.read_csv("ground_truth.csv", sep=",")
	#works for S_Godille_Table_1, S_Godille_Table_2, S_Godille_Table_3
	dt = pd.read_csv("/home/theoguidoux/INSA/ws/projetsecu4a/docs/CSV_RENDU/S_Godille_Table_1.csv", sep=",")

	print("Converting datasets...")
	#convert it to numpy array to speed up things
	gtn = np.asarray(gt)
	dtn = np.asarray(dt)

	#shapes
	#print(gtn.shape) #(307054, 6)
	#print(dtn.shape) #(314024, 6)

	#mega scaler of the death
	sscaler = StandardScaler()

	#print(gtn[0]) #[17850 '2010/12/01' '08:26' '85123A' 2.55 6]
	#print(dtn[0]) #[830706 '2011/11/01' '00:00' '23219' 2.001 1]

	print("Learning transformation from ground_truth and scalarizing inputs...")
	#learn how to convert str items to int items and store it in tables for date/hour/item
	trans_table_date, trans_table_date_rev, trans_table_hours, trans_table_hours_rev, trans_table_item, trans_table_item_rev, Xgt = learn_panda_to_vector(gt, sscaler)
	#print(Xgt.shape) #(307054, 6)
	#print(Xgt[0]) #[ 1.46934958 -0.83609854 -0.35887791  0.47985661 -0.02553646 -0.15785763]

	print("Applying transformation to anonymized dataset...")
	#now apply learnt vectorization on anonymized data
	dtn, trans_table_date, trans_table_date_rev = apply_transform(dtn, trans_table_date, 1)
	dtn, trans_table_hours, trans_table_hours_rev = apply_transform(dtn, trans_table_hours, 2)
	dtn, trans_table_item, trans_table_item_rev = apply_transform(dtn, trans_table_item, 3)

	print("Applying scalarization to anonymized dataset...")
	Xdt = sscaler.transform(dtn)

	#print(Xdt.shape) #(314024, 6)
	#print(Xdt[0]) #[ 4.71031071e+02 -1.26383084e-01  2.05117891e+00 -1.25426197e+00 -4.89814730e-02 -2.76827083e-01]

	print("Getting similar vector ...")
	#to test : change dtn[0] to gtn[0] and Xdt[0] to Xgt[0]

	rl = 10
	vec_index = 0

	print(gtn[vec_index])
	print(dtn[vec_index])
	sim_vectors, sim_scores = get_similar(Xgt, Xdt[vec_index], return_length=rl)
	for i in range(0, rl):
		sim_vec = sim_vectors[i]
		sim_score = sim_scores[i]
		rev_sim_vec = reverse_vector(sim_vec, sscaler, trans_table_date_rev, trans_table_hours_rev, trans_table_item_rev)
		print("(d = "+str(sim_score)+")", rev_sim_vec)


if __name__ == "__main__":
    main()
