
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import pandas as pd
import time, random, sys, os, argparse
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from threading import Thread

#turnn off fucking ugly warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from desanotools import *

gt_file_path = "ground_truth.csv"
desano_file_path = "S_Files/S_sub2_cat1narvali_desanonymised.csv"
s_file_path = "S_Files/S_sub2_cat1narvali.csv"
f_file_path = "F_Files/F_sub2_cat1narvali_by_cat1narvali.csv"

gt = pd.read_csv(gt_file_path, sep=",")
df = pd.read_csv(desano_file_path, sep=",")
sf = pd.read_csv(s_file_path, sep=",")

print("Converting datasets...")
gtn = np.asarray(gt)
dfn = np.asarray(df)
sfn = np.asarray(sf)

print("Splitting dates...")
gtn = split_date(gtn, 1)
dfn = split_date(dfn, 1)
sfn = split_date(sfn, 1)

print(gtn[0])
print(dfn[0])
print(sfn[0])

all_gt_id_user = list(set(gtn[:,0].astype(str).tolist()))
id_user_set = sorted(list(set(dfn[:,0].astype(str).tolist())))

#contains rows like "id_user,0,1,2,3,4,5,6,7,8,9,10,11,12"
f_file = open(f_file_path, "a")
f_file.write("id_user,0,1,2,3,4,5,6,7,8,9,10,11,12\n")

def get_best_id_freq(vecs):
	all_ids = vecs[:,0].tolist()
	all_ids_set = list(set(vecs[:,0].tolist()))
	freq_id = dict()
	for id in all_ids_set:
		freq_id[id] = all_ids.count(id)

	keys = list(freq_id.keys())
	values = list(freq_id.values())
	if len(keys) > 0:
		best_id = keys[values.index(max(values))]
		return best_id
	return "DEL"

print("Found", len(list(id_user_set)))
res = []
for id_user in id_user_set:

	all_dfn_rows_having_same_dfn_id_user = dfn[np.where(dfn[:, 0] == id_user)]
	all_sfn_rows_having_same_dfn_id_user = sfn[np.where(dfn[:, 0] == id_user)]
	#print(all_dfn_rows_having_same_dfn_id_user)
	#print(all_sfn_rows_having_same_dfn_id_user)
	#print(all_dfn_rows_having_same_dfn_id_user[:, 1])

	#all of the are in 0
	all_dfn_row_in_tendec = all_sfn_rows_having_same_dfn_id_user[np.where(all_sfn_rows_having_same_dfn_id_user[:, 1] == '2010')]
	best_tendec_id = get_best_id_freq(all_dfn_row_in_tendec)
	all_dfn_row_in_eledec = all_sfn_rows_having_same_dfn_id_user[np.where(all_sfn_rows_having_same_dfn_id_user[:, 1] == '2011')]

	all_dfn_row_in_eledecone = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '1')]
	all_dfn_row_in_eledectwo = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '2')]
	all_dfn_row_in_eledecthr = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '3')]
	all_dfn_row_in_eledecfou = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '4')]
	all_dfn_row_in_eledecfiv = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '5')]
	all_dfn_row_in_eledecsix = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '6')]
	all_dfn_row_in_eledecsev = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '7')]
	all_dfn_row_in_eledeceig = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '8')]
	all_dfn_row_in_eledecnin = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '9')]
	all_dfn_row_in_eledecten = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '10')]
	all_dfn_row_in_eledecele = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '11')]
	all_dfn_row_in_eledectwe = all_dfn_row_in_eledec[np.where(all_dfn_row_in_eledec[:, 2] == '12')]

	eledecone_best_id = get_best_id_freq(all_dfn_row_in_eledecone)
	eledectwo_best_id = get_best_id_freq(all_dfn_row_in_eledectwo)
	eledecthr_best_id = get_best_id_freq(all_dfn_row_in_eledecthr)
	eledecfou_best_id = get_best_id_freq(all_dfn_row_in_eledecfou)
	eledecfiv_best_id = get_best_id_freq(all_dfn_row_in_eledecfiv)
	eledecsix_best_id = get_best_id_freq(all_dfn_row_in_eledecsix)
	eledecsev_best_id = get_best_id_freq(all_dfn_row_in_eledecsev)
	eledeceig_best_id = get_best_id_freq(all_dfn_row_in_eledeceig)
	eledecnin_best_id = get_best_id_freq(all_dfn_row_in_eledecnin)
	eledecten_best_id = get_best_id_freq(all_dfn_row_in_eledecten)
	eledecele_best_id = get_best_id_freq(all_dfn_row_in_eledecele)
	eledectwe_best_id = get_best_id_freq(all_dfn_row_in_eledectwe)

	vec = [str(id_user), best_tendec_id, eledecone_best_id, eledectwo_best_id, eledecthr_best_id, eledecfou_best_id, eledecfiv_best_id, eledecsix_best_id, eledecsev_best_id, eledeceig_best_id, eledecnin_best_id, eledecten_best_id, eledecele_best_id, eledectwe_best_id]
	res.append(vec)

missing_ids = list(set(all_gt_id_user)-set([v[0] for v in res]))

for missing_id_user in missing_ids:
	vec = [str(missing_id_user), "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL"]
	res.append(vec)

res = sorted(res, key = lambda x: x[0])

for vec in res:
	#print(vec)
	f_file.write(','.join(vec)+"\n")
f_file.close()
