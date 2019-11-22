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

#do not change this values
id_index = 0
date_index = 1
hours_index = 2
item_index = 3
price_index = 4
qtt_index = 5

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
			new_entries_values = list(set(list(range(0, 3*len(trans_table.keys()))))-set(trans_table.values()))
			new_entry_value = random.choice(new_entries_values)
			trans_table[dfn[i][num_col]] = new_entry_value

		#apply transform
		dfn[i][num_col]=trans_table[dfn[i][num_col]]

	#get reversible table
	trans_table_rev = {v: k for k, v in trans_table.items()}
	return dfn, trans_table, trans_table_rev

def get_similar(Xgt, Xdt_row, return_length=10):
	"""
	return closest rows of dtn from gtn
	"""
	############################
	init_Xgt = Xgt
	#reduce space search
	Xgt = Xgt.copy().astype(np.float64) #make a copy before reducing input
	Xgt_indexes = np.arange(Xgt.shape[0])
	Xdt_row = Xdt_row.astype(np.float64) #convert to float 64 because Xgt are float64

	if force_year_equality:
		valid_rows = np.where(Xgt[:, date_index] == Xdt_row[date_index])
		Xgt = Xgt[valid_rows]
		Xgt_indexes = Xgt_indexes[valid_rows]
		del valid_rows
		#print(Xgt.shape)

	if force_month_equality:
		valid_rows = np.where(Xgt[:, date_index+1] == Xdt_row[date_index+1])
		Xgt = Xgt[valid_rows]
		Xgt_indexes = Xgt_indexes[valid_rows]
		del valid_rows
		#print(Xgt.shape)

	if force_day_equality:
		valid_rows = np.where(Xgt[:, date_index+2] == Xdt_row[date_index+2])
		Xgt = Xgt[valid_rows]
		Xgt_indexes = Xgt_indexes[valid_rows]
		del valid_rows
		#print(Xgt.shape)

	if force_item_equality:
		valid_rows = np.where(Xgt[:, item_index] == Xdt_row[item_index])
		Xgt = Xgt[valid_rows]
		Xgt_indexes = Xgt_indexes[valid_rows]
		del valid_rows
		#print(Xgt.shape)

	if force_qtt_equality:
		valid_rows = np.where(Xgt[:, qtt_index] == Xdt_row[qtt_index])
		Xgt = Xgt[valid_rows]
		Xgt_indexes = Xgt_indexes[valid_rows]
		del valid_rows
		#print(Xgt.shape)

	#print("Space search reduced to",Xgt.shape[0],"possibilities")

	similar_rows = []
	similar_rows_score = []

	if Xgt.shape[0] > 0:

		#print(cosine_similarity([Xgt[0]], [Xdt_row])[0][0]) #return the similarity of Xgt[0] and Xdt_row => 0.5008987774997233
		#print(cosine_similarity([Xgt[0]], [Xgt[0]])[0][0]) #return the similarity of Xgt[0] and Xgt[0] => 1.0
		#get distance to origin euclidean_distances([[0, 1], [1, 1]], [[0, 0]]) => array([[1.], [1.41421356]])
		#similarities = sorted(enumerate([cosine_similarity([Xgt_row], [Xdt_row])[0][0] for Xgt_row in Xgt]), key = lambda x: int(x[1]))
		eds = euclidean_distances(Xgt, [Xdt_row])
		#decrease dimension
		eds = [ed[0] for ed in eds]

		#eds are euclidean_distances from Xgt to Xdt_row
		#Xgt_indexes are corresponding indexes of each row in init_Xgt
		#so we stack them and sort them following eds to have best indexes in init_Xgt
		eds = np.asarray(eds)
		stacked = np.stack((Xgt_indexes, eds), axis=-1) #[[Xgt_index, score], [Xgt_index, score], [Xgt_index, score], ...]

		#list of distances from ground_truth vectors to Xdt_row sorted from lower to higher with corresponding index in Xgt
		distances = sorted(stacked.tolist(), key = lambda x: x[1])
		distances = [(int(d[0]), d[1]) for d in distances]
		#print("max distance : ", max([d[1] for d in distances]))
		#print("min distance : ", min([d[1] for d in distances]))

		for i in range(0, return_length):
			#(Xgt index, Xgt_vect)
			#euclidian distance / score
			similar_rows.append((distances[i][0], init_Xgt[distances[i][0]]))
			similar_rows_score.append(distances[i][1])

		del distances, eds, stacked

	del Xgt, Xgt_indexes

	return np.asarray(similar_rows), similar_rows_score

def data_to_vector(data, scaler, trans_table_date_y, trans_table_date_m, trans_table_date_d, trans_table_hours, trans_table_item):
	dtn = np.asarray([data])
	if use_dates:
		dtn, _, _ = apply_transform(dtn, trans_table_date_y, date_index)
		dtn, _, _ = apply_transform(dtn, trans_table_date_m, date_index+1)
		dtn, _, _ = apply_transform(dtn, trans_table_date_d, date_index+2)

	if use_hours:
		dtn, _, _ = apply_transform(dtn, trans_table_hours, hours_index)

	if use_index:
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
	if use_dates:
		unscaled_vec[date_index] = trans_table_date_y_rev[int(unscaled_vec[date_index])]
		unscaled_vec[date_index+1] = trans_table_date_m_rev[int(unscaled_vec[date_index+1])]
		unscaled_vec[date_index+2] = trans_table_date_d_rev[int(unscaled_vec[date_index+2])]

	if use_hours:
		unscaled_vec[hours_index] = trans_table_hours_rev[int(unscaled_vec[hours_index])]
	if use_items:
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
	global id_index, date_index, item_index, hours_index, price_index, qtt_index, use_scaler

	############################################################################
	#read and convert datasets
	#gt is the ground_truth dataset and dt is the anonymized dataset to crack (dt for data)
	print("Reading datasets...")

	#works for S_Godille_Table_1, S_Godille_Table_2, S_Godille_Table_3
	gt = pd.read_csv("ground_truth.csv", sep=",")
	dt = pd.read_csv("/home/theoguidoux/INSA/ws/projetsecu4a/docs/CSV_RENDU/S_Godille_Table_3.csv", sep=",")

	print("Converting datasets...")
	gtn = np.asarray(gt)
	dtn = np.asarray(dt)
	#[17850 '2010/12/01' '08:26' '85123A' 2.55 6]

	############################################################################
	#delete axis in function of what we want to do : check data_index, item_index, hours_index
	print("Deleting useless columns...")
	if not use_index:
		gtn = np.delete(gtn, id_index, axis=1)
		dtn = np.delete(dtn, id_index, axis=1)
		id_index = id_index - 1
		date_index = date_index - 1
		hours_index = hours_index - 1
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	if not use_dates:
		gtn = np.delete(gtn, date_index, axis=1)
		dtn = np.delete(dtn, date_index, axis=1)
		date_index = date_index - 1
		hours_index = hours_index - 1
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	if not use_hours:
		gtn = np.delete(gtn, hours_index, axis=1)
		dtn = np.delete(dtn, hours_index, axis=1)
		hours_index = hours_index - 1
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	if not use_items:
		gtn = np.delete(gtn, item_index, axis=1)
		dtn = np.delete(dtn, item_index, axis=1)
		item_index = item_index - 1
		price_index = price_index - 1
		qtt_index = qtt_index - 1

	############################################################################
	#if we use date_index, so we convert XX/XX/XXXX to XX, XX, XX and ad it to final vector
	if use_dates:
		print("Splitting dates...")
		gtn = split_date(gtn, date_index)
		dtn = split_date(dtn, date_index)
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

	trans_table_date_y, trans_table_date_y_rev,\
	trans_table_date_m,trans_table_date_m_rev,\
	trans_table_date_d, trans_table_date_d_rev,\
	trans_table_hours, trans_table_hours_rev,\
	trans_table_item, trans_table_item_rev,\
	Xgt = learn_panda_to_vector(gtn.copy(), sscaler)

	#############################
	#step 2 : apply transformations on anonymized data so that
	#with this transformations ground_truth and anonymized data can be compared
	print("Applying transformation on anonymized dataset...")

	#keep a clean copy of dtn
	dtn_transformed = dtn.copy()

	if use_dates:
		dtn_transformed, trans_table_date_y, trans_table_date_y_rev = apply_transform(dtn_transformed, trans_table_date_y, date_index)
		dtn_transformed, trans_table_date_m, trans_table_date_m_rev = apply_transform(dtn_transformed, trans_table_date_m, date_index+1)
		dtn_transformed, trans_table_date_d, trans_table_date_d_rev = apply_transform(dtn_transformed, trans_table_date_d, date_index+2)

	if use_hours:
		dtn_transformed, trans_table_hours, trans_table_hours_rev = apply_transform(dtn_transformed, trans_table_hours, hours_index)

	if use_items:
		dtn_transformed, trans_table_item, trans_table_item_rev = apply_transform(dtn_transformed, trans_table_item, item_index)

	if use_scaler:
		print("Applying scalarization on anonymized dataset...")
		Xdt = sscaler.transform(dtn_transformed)
	else:
		Xdt = dtn_transformed

	Xgt = np.asarray(Xgt, dtype=np.float64)
	Xdt = np.asarray(Xdt, dtype=np.float64)

	############################################################################
	#just to make things clear

	#Xgt and Xdt are vectors based on ground_truth and converted to float64 numbers
	#print(Xgt[0]) #[0.000e+00 1.100e+01 2.600e+01 3.259e+03 2.550e+00 6.000e+00]
	#print(Xdt[0]) #[1.000e+00 8.000e+00 2.600e+01 1.570e+03 2.001e+00 1.000e+00]

	#gtn and dtn are data without useless columns but not vectorized
	#print(gtn[0]) #['2010' '12' '1' '85123A' '2.55' '6']
	#print(dtn[0]) #['2011' '11' '1' '23219' '2.001' '1']

	#gt and dt are data with all columns, not vectorized (input data)
	#print(list(gt.loc[0])) #[17850, '2010/12/01', '08:26', '85123A', 2.55, 6]
	#print(list(dt.loc[0])) #[830706, '2011/11/01', '00:00', '23219', 2.001, 1]

	############################################################################
	#find the closest vector to Xdt[i] in Xgt
	print("Let's hack now !!!")

	nb_result = 1 			#change this to have more result
	result = []
	for i in range(0, 5):	#dtn_transformed.shape[0] change the 5 to dtn_transformed.shape[0] to run all anonymized data
		print("Computing "+str(i+1)+" out of "+str(Xdt.shape[0]), end="\r")

		input_data = dtn_transformed[i]	#the vectorized data we want to crack

		#sim_vectors and im_scores contains are list of size <nb_result> having closest vectors of input_vec from Xgt
		sim_vectors, sim_scores = get_similar(Xgt, input_data, return_length=nb_result)
		result.append((i, input_data.tolist(), sim_vectors, sim_scores))

	############################################################################
	#pretty print result

	list_desanonymized = []

	print("\n\n\n")
	print("Results :")
	for res in result:
		dt_vec_index = res[0]			#index of the row in dtn (this is the index of the row that we want to crack)
		dt_vec = res[1]					#the row that we want to crack which is dtn_transformed[i]
		closest_vecs = res[2]			#this is a list of tuple (index_of_closest_row_in_Xgt, closest_Xgt_row)
		scores = res[3]					#list of score corresponding to the tuple of closest_vecs having close score of closest_vec
		for i in range(0, len(closest_vecs)):
			gt_closest_entry_index = closest_vecs[i][0]						#index_of_closest_row_in_Xgt
			closest_vec = closest_vecs[i][1]								#closest_Xgt_row
			score = scores[i]												#close score of closest_vec
			gt_closest_entry = np.asarray(gt.loc[gt_closest_entry_index])	#the closest row in gt mean not vectorized (the result in clear of the cracked row)
			list_desanonymized.append(gt_closest_entry)
			dt_entry = np.asarray(dt.loc[dt_vec_index])						#the row that is cracekd not vectorized

			#closest_entry is the same as gt_closest_entry but gt_closest_entry has the user_id
			#closest_entry = reverse_vector(closest_vec, sscaler, trans_table_date_y_rev, trans_table_date_m_rev, trans_table_date_d_rev, trans_table_hours_rev, trans_table_item_rev)

			#result is displayed here
			#print("#"*50)
			#print("Closest vector of anonymized row (dtn index = "+str(dt_vec_index)+")", dt_entry, "is", gt_closest_entry, "(gtn index = "+str(gt_closest_entry_index)+")")
			#print("data :", dt_entry, "=>", gt_closest_entry)
			#print("vectors :", dt_vec, "=>", list(closest_vec))
			#print("score =", score)

	dt_desanonymized = pd.DataFrame(list_desanonymized, columns=["id_user","date","hours","id_item","price","qty"])
	dt_desanonymized.to_csv("dt_desanonymized.csv", index=False)

if __name__ == "__main__":
	main()
