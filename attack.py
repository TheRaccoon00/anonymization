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
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

#turnn off fucking ugly warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#best values :
#date_index = 0
#hours_index = -1
#item_index = 1
#use_scaler = False

date_index = 0
hours_index = -1
item_index = 1
use_scaler = False

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
	if date_index != -1:
		print("Transforming date")
		trans_table_date_y, trans_table_date_y_rev, X = vectorize(X, date_index)
		trans_table_date_m, trans_table_date_m_rev, X = vectorize(X, date_index+1)
		trans_table_date_d, trans_table_date_d_rev, X = vectorize(X, date_index+2)


	if hours_index != -1:
		print("Transforming hours")
		trans_table_hours, trans_table_hours_rev, X = vectorize(X, hours_index)

	if item_index != -1:
		print("Transforming id_item")
		trans_table_item, trans_table_item_rev, X = vectorize(X, item_index)

	#print(X[0]) => [17850 0 10 241 2.55 6]

	#apply scaler to input
	if use_scaler:
		X = scaler.fit_transform(X)
	return 	trans_table_date_y, trans_table_date_y_rev,\
			trans_table_date_m, trans_table_date_m_rev,\
			trans_table_date_d, trans_table_date_d_rev,\
			trans_table_hours, trans_table_hours_rev, trans_table_item, trans_table_item_rev, X

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
		similar_rows.append((distances[i][0], Xgt[distances[i][0]]))
		similar_rows_score.append(distances[i][1])

	return np.asarray(similar_rows), similar_rows_score

def data_to_vector(data, scaler, trans_table_date_y, trans_table_date_m, trans_table_date_d, trans_table_hours, trans_table_item):
	dtn = np.asarray([data])
	if date_index != -1:
		dtn, _, _ = apply_transform(dtn, trans_table_date_y, date_index)
		dtn, _, _ = apply_transform(dtn, trans_table_date_m, date_index+1)
		dtn, _, _ = apply_transform(dtn, trans_table_date_d, date_index+2)

	if hours_index != -1:
		dtn, _, _ = apply_transform(dtn, trans_table_hours, hours_index)

	if item_index != -1:
		dtn, _, _ = apply_transform(dtn, trans_table_item, item_index)

	transformed = dtn
	if use_scaler:
		transformed = scaler.transform(dtn)
	return transformed[0]

def reverse_vector(vec, scaler, trans_table_date_y_rev, trans_table_date_m_rev, trans_table_date_d_rev, trans_table_hours_rev, trans_table_item_rev):
	if use_scaler:
		unscaled_vec = list(scaler.inverse_transform([vec])[0])
	else:
		unscaled_vec = vec
	if date_index != -1:
		unscaled_vec[date_index] = trans_table_date_y_rev[int(unscaled_vec[date_index])]
		unscaled_vec[date_index+1] = trans_table_date_m_rev[int(unscaled_vec[date_index+1])]
		unscaled_vec[date_index+2] = trans_table_date_d_rev[int(unscaled_vec[date_index+2])]

	if hours_index != -1:
		unscaled_vec[hours_index] = trans_table_hours_rev[int(unscaled_vec[hours_index])]
	if item_index != -1:
		unscaled_vec[item_index] = trans_table_item_rev[int(unscaled_vec[item_index])]

	return unscaled_vec

def split_date(X, date_index):
	out = []
	for i in range(0, X.shape[0]):
		y, m, d = X[i][date_index].split("/")
		y, m, d = int(y), int(m), int(d)
		x_i = list(X[i])
		x_i[date_index:date_index+1] = [y, m, d]
		out.append(x_i)
	return np.asarray(out)

def main():
	global date_index, item_index, hours_index, use_scaler
	print("Reading datasets...")
	#load gt and dt is the anonymized dataset
	gt = pd.read_csv("ground_truth.csv", sep=",")
	#works for S_Godille_Table_1, S_Godille_Table_2, S_Godille_Table_3
	dt = pd.read_csv("/home/theoguidoux/INSA/ws/projetsecu4a/docs/CSV_RENDU/S_Godille_Table_3.csv", sep=",")

	print("Converting datasets...")
	#convert it to numpy array to speed up things
	gtn = np.asarray(gt)
	dtn = np.asarray(dt)
	#[17850 '2010/12/01' '08:26' '85123A' 2.55 6]

	#delete some axes id and hours
	gtn = np.delete(gtn, [0, 2], axis=1)
	dtn = np.delete(dtn, [0, 2], axis=1)


	if date_index != -1:
		print("Splitting dates...")
		#split date
		gtn = split_date(gtn, date_index)
		dtn = split_date(dtn, date_index)
		item_index = item_index + 2 #deleted 1 item, added 3 so add 2 to item_index

	#shapes
	#print(gtn.shape) #(307054, 6)
	#print(dtn.shape) #(314024, 6)

	#mega scaler of the death
	#sscaler = StandardScaler()
	sscaler = MaxAbsScaler()
	#print(gtn[0]) #[17850 '2010/12/01' '08:26' '85123A' 2.55 6]
	#print(dtn[0]) #[830706 '2011/11/01' '00:00' '23219' 2.001 1]

	print("Learning transformation from ground_truth and scalarizing inputs...")
	#learn how to convert str items to int items and store it in tables for date/hour/item

	trans_table_date_y, trans_table_date_y_rev,\
	trans_table_date_m,trans_table_date_m_rev,\
	trans_table_date_d, trans_table_date_d_rev,\
	trans_table_hours, trans_table_hours_rev,\
	trans_table_item, trans_table_item_rev,\
	Xgt = learn_panda_to_vector(gtn, sscaler)

	#print(Xgt.shape) #(307054, 6)
	#print(Xgt[0]) #[ 1.46934958 -0.83609854 -0.35887791  0.47985661 -0.02553646 -0.15785763]

	print("Applying transformation to anonymized dataset...")
	#now apply learnt vectorization on anonymized data
	dtn_transformed = dtn.copy()
	if date_index != -1:
		dtn_transformed, trans_table_date_y, trans_table_date_y_rev = apply_transform(dtn_transformed, trans_table_date_y, date_index)
		dtn_transformed, trans_table_date_m, trans_table_date_m_rev = apply_transform(dtn_transformed, trans_table_date_m, date_index+1)
		dtn_transformed, trans_table_date_d, trans_table_date_d_rev = apply_transform(dtn_transformed, trans_table_date_d, date_index+2)

	if hours_index != -1:
		dtn_transformed, trans_table_hours, trans_table_hours_rev = apply_transform(dtn_transformed, trans_table_hours, hours_index)

	if item_index != -1:
		dtn_transformed, trans_table_item, trans_table_item_rev = apply_transform(dtn_transformed, trans_table_item, item_index)


	print("Applying scalarization to anonymized dataset...")
	if use_scaler:
		Xdt = sscaler.transform(dtn_transformed)
	else:
		Xdt = dtn_transformed
	Xgt = np.asarray(Xgt, dtype=np.float64)
	Xdt = np.asarray(Xdt, dtype=np.float64)
	print(Xgt[0])
	print(Xdt[0])
	#print(Xdt.shape) #(314024, 6)
	#print(Xdt[0]) #[ 4.71031071e+02 -1.26383084e-01  2.05117891e+00 -1.25426197e+00 -4.89814730e-02 -2.76827083e-01]

#	print("Getting similar vector ...")
#	#to test : change dtn[0] to gtn[0] and Xdt[0] to Xgt[0]
#
#	rl = 10
#	vec_index = 0
#
#	print(gtn[vec_index])
#	print(dtn[vec_index])
#	sim_vectors, sim_scores = get_similar(Xgt, Xdt[vec_index], return_length=rl)
#	for i in range(0, rl):
#		sim_vec = sim_vectors[i]
#		sim_score = sim_scores[i]
#		rev_sim_vec = reverse_vector(sim_vec, sscaler, trans_table_date_rev, trans_table_hours_rev, trans_table_item_rev)
#		print("(d = "+str(sim_score)+")", rev_sim_vec)

	print("Getting similar vectors ...")
	nb_result = 1 #change this to have more result
	result = []
	for i in range(0, 5):#dtn.shape[0]
		print("Computing "+str(i+1)+" out of "+str(Xdt.shape[0]), end="\r")
		##print(str(round((i*100)/Xdt.shape[0], 3))+"%", end="\r")
		input_data = dtn[i]
		input_vec = data_to_vector(input_data, sscaler, trans_table_date_y, trans_table_date_m, trans_table_date_d, trans_table_hours, trans_table_item)
		#find the closest vector to Xdt[i] in Xgt
		sim_vectors, sim_scores = get_similar(Xgt, input_vec, return_length=nb_result)
		result.append((i, input_data.tolist(), sim_vectors, sim_scores))

	#print result
	print("\n\n\n")
	print("Results :")
	for res in result:
		anonymized_entry_index = res[0]
		anonymized_entry = res[1]
		closest_vecs = res[2]
		scores = res[3]
		for i in range(0, len(closest_vecs)):
			init_closest_vec_index = closest_vecs[i][0]
			closest_vec = closest_vecs[i][1]
			score = scores[i]
			init_closest_vec = np.asarray(gt.loc[init_closest_vec_index])
			init_anonymized_vec = np.asarray(dt.loc[anonymized_entry_index])
			closest_entry = reverse_vector(closest_vec, sscaler, trans_table_date_y_rev, trans_table_date_m_rev, trans_table_date_d_rev, trans_table_hours_rev, trans_table_item_rev)
			print("#"*50)
			print("data :", init_anonymized_vec, "=>", init_closest_vec)
			print("vectors :", anonymized_entry, "=>", closest_entry)
			print("score =", score, " == ? :", anonymized_entry == closest_entry)

if __name__ == "__main__":
	print("################################################")
	print("## Add similarity rules like quantity exactitude cause qtty may not be anonymized")
	print("################################################")
	main()
