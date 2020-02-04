"""
File: attack.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database anonymizer for DARC competition
"""

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import pandas as pd
import time, random, sys, os, argparse, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from threading import Thread
from encoder import AutoEncoderTrainer

#turnn off fucking ugly warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from encoder import load_autoencoder
from desanotools import *
from plotter import *

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

gt_path = ""		#ground truth file path
dt_path = ""		#anonymized file path
out_path = ""		#output file path
conf_file_path = ""

encoder_model = load_autoencoder("encoder.h5")

def hack(conf, id_user, gt, Xgt, dtn_transformed_part, nb_result, result):
	#retourne l'id_user le plus probable de la liste de course dtn_transformed_part
	#part_result = []
	#print("dtn_transformed_part.shape[0]", dtn_transformed_part.shape[0])
	#for i in range(0, dtn_transformed_part.shape[0]):	#dtn_transformed.shape[0] change the 5 to dtn_transformed.shape[0] to run all anonymized data
	#print("\t"*(3*index)+"[Thread"+str(index)+"]"+str(i+1)+"/"+str(dtn_transformed_part.shape[0]), end="\r")
	#print("hacking "+id_user+"\t\t| Xgt rows : "+str(Xgt.shape[0]))#, end="\r")
	sim_vectors = []
	sim_scores = []

	for i in range(0, dtn_transformed_part.shape[0]):
		#print("\t"*(3*index)+"[Thread"+str(index)+"]"+str(i+1)+"/"+str(dtn_transformed_part.shape[0]), end="\r")
		#sim_vectors and sim_scores are list of size <nb_result> having closest vectors of dtn_transformed_part from Xgt
		sim_vectors_rows, sim_scores_rows = get_similar(gt, Xgt, dtn_transformed_part[i], conf, encoder_model, return_length=nb_result)
		[sim_vectors.append(sim_vectors_row) for sim_vectors_row in sim_vectors_rows]
		[sim_scores.append(sim_scores_row) for sim_scores_row in sim_scores_rows]

		#part_result.append((i, input_data.tolist(), sim_vectors, sim_scores))
	#print("sim_vectors", sim_vectors)
	#print("list(gt.loc[sim_vectors[i][0]])", list(gt.loc[sim_vectors[i][0]]))
	#[part_result.append(list(gt.loc[sim_vectors[i][0]])) for i in range(0, len(sim_vectors))]
	#print("gt", gt)
	st = time.time()
	id_user_freq = dict()
	for res in sim_vectors:
		id_user_v = gt.loc[res[0]][0]
		#print(id_user)
		if id_user_freq.get(str(id_user_v), None) == None:
			id_user_freq[str(id_user_v)] = 1
		else:
			id_user_freq[str(id_user_v)] += 1

	for k in set(id_user_freq.keys()).intersection(set(result.keys())):
		id_user_freq[k] -= 1

	#print(id_user_freq)
	id_user_freq = {k: v for k, v in sorted(id_user_freq.items(), key=lambda item: item[1], reverse=True)}
	#print(id_user_freq)
	k = list(id_user_freq.keys())
	if len(k) > 0:
		best_desanonymised_id_user = k[0]
	else:
		return "DEL"
	#print("")
	#print(id_user, "=>", best_desanonymised_id_user)
	#exit()
	#print("frequency analysis duration =", time.time()-st)
	return best_desanonymised_id_user

def main():
	global id_index, date_index, item_index, hours_index, price_index, qtt_index, use_scaler, nb_threads, encoder_model
	############################################################################
	#read and convert datasets
	#gt is the ground_truth dataset and dt is the anonymized dataset to crack (dt for data)
	print("Reading datasets...")

	#works for S_Godille_Table_1, S_Godille_Table_2, S_Godille_Table_3
	gt = pd.read_csv(gt_path, sep=",")
	dt = pd.read_csv(dt_path, sep=",")

	print("Converting datasets...")
	gtn = np.asarray(gt)
	dtn = np.asarray(dt)
	#[17850 '2010/12/01' '08:26' '85123A' 2.55 6]

	print("Resolving shopping lists...")
	shopping_lists = find_shopping_list(dtn, id_index)

	gtn_init = gtn.copy()
	rows_index_to_delete = []
	gtn_shopping_lists = find_shopping_list(gtn_init, 0)
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

	conf = compact_conf(use_index, use_dates, use_hours, use_items, use_scaler, force_year_equality, force_month_equality, force_day_equality, force_item_equality, force_qtt_equality, nb_threads, show_result, id_index, date_index, hours_index, item_index, price_index, qtt_index, gt_path, dt_path, out_path)
	save_conf(conf, conf_file_path)

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
	Xgt = learn_panda_to_vector(gtn.copy(), sscaler, conf)

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
	Xdt	 = np.asarray(Xdt, dtype=np.float64)

	############################################################################
	#make autoencoder easy to train
	need_autoencoder_train = input("Need autoencoder train ? (y/*) : ")
	if need_autoencoder_train.lower() == "y":
		aet = AutoEncoderTrainer(Xgt)

		train_successful = False
		while not train_successful:
			aet.train()
			train_successful_input = input("Train successful ? (y/*) : ")
			if train_successful_input.lower() == "y":
				train_successful = True

		aet.save_model(aet.encoder_model, "encoder.h5")

		encoder_model = load_autoencoder("encoder.h5")

	############################################################################
	#it's funny to see data in 3d !
	#plot_db(Xgt, Xdt, encoder_model)

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
	#/!\ we don't use threading anymore

	print("Let's hack now !!!")

	nb_result = 75			#change this to have more result, default 1
	result = dict()			#final main result handler

	print("Found", len(list(shopping_lists.keys())), "users to desanonymize")
	print("")
	for i, id_user in enumerate(shopping_lists.keys()):

		#list of rows from gtn with id = id_user
		items_transformed = np.asarray([dtn_transformed[index] for index in [sl[0] for sl in shopping_lists[id_user]]])
		#print("id_user", id_user)
		#print("items_transformed", items_transformed)
		new_Xgt_with_deleted_items = Xgt#np.delete(Xgt, rows_index_to_delete, 0)
		#print("shape", new_Xgt_with_deleted_items.shape)
		#print("rows_index_to_delete", rows_index_to_delete)
		#print("new_Xgt_with_deleted_items", new_Xgt_with_deleted_items)
		st = time.time()
		best_desanonymised_id_user = hack(conf, id_user, gt, Xgt, items_transformed, nb_result, result)
		result[id_user] = best_desanonymised_id_user

		print("\n=>", id_user, "=>", best_desanonymised_id_user, "|", len(list(shopping_lists.keys()))-i-1, "/", len(list(shopping_lists.keys()))-1,"id_users remaining, ")
		print("hack duration =", time.time()-st, "(s) | len =>", items_transformed.shape[0])
		#print("gtn_shopping_lists[best_desanonymised_id_user]", gtn_shopping_lists[best_desanonymised_id_user])
		#print(gtn_shopping_lists[best_desanonymised_id_user])
		#rows_index_to_delete = rows_index_to_delete#+[sl[0] for sl in gtn_shopping_lists[best_desanonymised_id_user]]

	############################################################################
	#pretty print result
	print("Saving result...")

	#dt_desanonymized = pd.DataFrame(list_desanonymized, columns=["id_user","date","hours","id_item","price","qty"])
	#dt_desanonymized.to_csv(out_path, index=False)
	#gtn_export = split_date(np.asarray(gt), 1)
	#sfn_export = split_date(np.asarray(dt), 1)
	#export_f_file(gtn_export, sfn_export, np.asarray(list_desanonymized), out_path)

	#print(result)
	with open(os.path.splitext(out_path)[0]+'.pickle', 'wb') as config_dictionary_file:
		pickle.dump({"gt":gt, "dt":dt, "result":result, "out_path":out_path}, config_dictionary_file)
	#output(gt, result, out_path)
	output2(gt, dt, result, os.path.splitext(out_path)[0]+"_v2.csv")

	save_conf(conf, conf_file_path)
	print("Result written in", out_path)
	print("Conf saved in ", conf_file_path)


if __name__ == "__main__":

	print("#########################################")
	print("# TODO : Réduire le temps de calcul (multithreading)")
	print("#########################################")

	parser = argparse.ArgumentParser()
	parser.add_argument("gt", help="ground_truth csv path", type=str)
	parser.add_argument("dt", help="anonymized csv path", type=str)

	parser.add_argument("-o", "--out", help="output csv file path", type=str, default="")

	parser.add_argument("-uin", "--use-index", help="use id_user for vectorization", default=False, action="store_true")
	parser.add_argument("-uda", "--use-dates", help="do not use date for vectorization", default=True, action="store_false")
	parser.add_argument("-uho", "--use-hours", help="use hour for vectorization", default=False, action="store_true")
	parser.add_argument("-uit", "--use-items", help="do not use id_item for vectorization", default=True, action="store_false")
	parser.add_argument("-usc", "--use-scaler", help="use scaler for vectorization", default=False, action="store_true")

	parser.add_argument("-fye", "--force-year-equality", help="force years to be equal when finding closest vectors", default=True, action="store_false")
	parser.add_argument("-fme", "--force-month-equality", help="force months to be equal when finding closest vectors", default=True, action="store_false")
	parser.add_argument("-fde", "--force-day-equality", help="do not force days to be equal when finding closest vectors", default=False, action="store_true")
	parser.add_argument("-fie", "--force-item-equality", help="force items to be equal when finding closest vectors", default=True, action="store_false")
	parser.add_argument("-fqe", "--force-qtt-equality", help="force qtts to be equal when finding closest vectors", default=True, action="store_false")

	parser.add_argument("-t", "--threads", help="number of threads", type=int, default=1)

	parser.add_argument("-v", "--verbose", help="show more informations", default=False, action="store_true")

	args = parser.parse_args()

	#define mandatory values
	gt_path = args.gt
	dt_path = args.dt

	conf_file_path = "desanoconf/conf.json"

	#define optional values
	out_path = args.out if args.out != "" else os.path.splitext(args.dt)[0]+"_desanonymised.csv"
	use_index = args.use_index						#default False
	use_dates = args.use_dates						#default True
	use_hours = args.use_hours						#default False
	use_items = args.use_items						#default True
	use_scaler = args.use_scaler					#default False

	force_year_equality = args.force_year_equality 	#default True
	force_month_equality = args.force_month_equality#default True
	force_day_equality = args.force_day_equality 	#default False
	force_item_equality = args.force_item_equality 	#default True
	force_qtt_equality = args.force_qtt_equality 	#default True

	#disabled for the v2 since we have to update gt
	nb_threads = args.threads						#default 1

	show_result = args.verbose

	main()
