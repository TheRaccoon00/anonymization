
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

def get_bought_item(id_user, xtn):
	bought_item = xtn[np.where(xtn[:, 0] == id_user)][:, 5]
	return bought_item

def get_shopping_list_sim_score(bi_u1, bi_u2):
	score = 0
	bl = bi_u1
	nbl = bi_u2

	if len(bi_u2) > len(bi_u1):
		 bl = bi_u2
		 nbl = bi_u1

	for item in bl:
		if item in nbl:
			score += 1

	score = score-abs(len(nbl)-len(bl))
	return score

def get_best_corresponding_gt_id_user(bi_dt_user, gt_id_user_set, bi_gt_dict):
	#bi_dt_user = get_bought_item(dt_id_user, dtn)
	max_score = -10000000000000000000000000
	best_gt_id_user = ""

	for gt_id_user in gt_id_user_set:
		bi_gt_user = bi_gt_dict[gt_id_user]
		st = time.time()
		score = get_shopping_list_sim_score(bi_gt_user, bi_dt_user)
		print(time.time()-st)
		if score > max_score:
			best_gt_id_user = gt_id_user
			max_score = score
	return max_score, best_gt_id_user

gt_path = "ground_truth.csv"
dt_path = "S_Files/S_stage3.csv"
out_path = "F_Files/F_stage3_desanonimysed.csv"

gt = pd.read_csv(gt_path, sep=",")
dt = pd.read_csv(dt_path, sep=",")

print("Converting datasets...")
gtn = np.asarray(gt)
dtn = np.asarray(dt)

#gtn = gtn[3456:4567].copy()
#print(gtn.shape)
#dtn = dtn[3456:4567].copy()
#print(dtn.shape)

print("Splitting dates...")
gtn = split_date(gtn, 1)
dtn = split_date(dtn, 1)

gt_id_user_set = sorted(list(set(gtn[:,0].tolist())))
dt_id_user_set = sorted(list(set(dtn[:,0].tolist())))

bi_gt_dict = dict()
bi_dt_dict = dict()

print("Resolving gt bought items...")
for i, gt_id_user in enumerate(gt_id_user_set):
	print(i,"/",len(gt_id_user_set), end="\r")
	bi_gt_dict[gt_id_user] = get_bought_item(gt_id_user, gtn)
#	if i == 100:
#		break

print("Resolving dt bought items...")
for i, dt_id_user in enumerate(dt_id_user_set):
	print(i,"/",len(dt_id_user_set), end="\r")
	bi_dt_dict[dt_id_user] = get_bought_item(dt_id_user, dtn)
#	if i == 100:
#		break

#print(bi_gt_dict[list(bi_gt_dict.keys())[0]])
#print(bi_dt_dict[list(bi_dt_dict.keys())[0]])
#print(len(list(bi_gt_dict.keys())), len(gt_id_user_set))
#print(len(list(bi_dt_dict.keys())), len(dt_id_user_set))

#list having object with [dt_id_user, corresponding gt_id_user]
result = []
for i, dt_id_user in enumerate(dt_id_user_set):
	print(i,"/",len(dt_id_user_set), "analysing", dt_id_user, end="\r")

	bi_dt_user = bi_dt_dict[dt_id_user]
	max_score, best_gt_id_user = get_best_corresponding_gt_id_user(bi_dt_user, gt_id_user_set, bi_gt_dict)
	#print(max_score, best_gt_id_user)
	#remove best_gt_id_user from gt_id_user_set because each of anonymised id have single corresponding gt_id
	gt_id_user_set.remove(best_gt_id_user)
	result.append([dt_id_user, best_gt_id_user])

result = sorted(result, key = lambda x: x[1])

for res in result:
	dt_id_user, gt_id_user = res

	dt_id_user_rows = dtn[np.where(dtn[:, 0] == dt_id_user)]
	gt_id_user_rows = gtn[np.where(gtn[:, 0] == gt_id_user)]

	#print(all_dfn_rows_having_same_dfn_id_user[:, 1])

	#all of them are in 0
	gt_id_user_rows_tendec = gt_id_user_rows[np.where(gt_id_user_rows[:, 1] == '2010')]

	gt_id_user_rows_in_eledec = gt_id_user_rows[np.where(gt_id_user_rows[:, 1] == '2011')]

	gt_id_user_rows_in_eledecone = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '1')]
	gt_id_user_rows_in_eledectwo = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '2')]
	gt_id_user_rows_in_eledecthr = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '3')]
	gt_id_user_rows_in_eledecfou = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '4')]
	gt_id_user_rows_in_eledecfiv = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '5')]
	gt_id_user_rows_in_eledecsix = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '6')]
	gt_id_user_rows_in_eledecsev = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '7')]
	gt_id_user_rows_in_eledeceig = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '8')]
	gt_id_user_rows_in_eledecnin = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '9')]
	gt_id_user_rows_in_eledecten = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '10')]
	gt_id_user_rows_in_eledecele = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '11')]
	gt_id_user_rows_in_eledectwe = gt_id_user_rows_in_eledec[np.where(gt_id_user_rows_in_eledec[:, 2].astype(np.int64) == '12')]

	best_tendec_id = "DEL" if gt_id_user_rows_tendec.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecone = "DEL" if gt_id_user_rows_in_eledecone.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledectwo = "DEL" if gt_id_user_rows_in_eledectwo.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecthr = "DEL" if gt_id_user_rows_in_eledecthr.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecfou = "DEL" if gt_id_user_rows_in_eledecfou.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecfiv = "DEL" if gt_id_user_rows_in_eledecfiv.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecsix = "DEL" if gt_id_user_rows_in_eledecsix.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecsev = "DEL" if gt_id_user_rows_in_eledecsev.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledeceig = "DEL" if gt_id_user_rows_in_eledeceig.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecnin = "DEL" if gt_id_user_rows_in_eledecnin.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecten = "DEL" if gt_id_user_rows_in_eledecten.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledecele = "DEL" if gt_id_user_rows_in_eledecele.shape[0] == 0 else dt_id_user
	gt_id_user_rows_in_eledectwe = "DEL" if gt_id_user_rows_in_eledectwe.shape[0] == 0 else dt_id_user

	print(gt_id_user, best_tendec_id, gt_id_user_rows_in_eledecone, gt_id_user_rows_in_eledectwo, gt_id_user_rows_in_eledecthr, gt_id_user_rows_in_eledecfou, gt_id_user_rows_in_eledecfiv, gt_id_user_rows_in_eledecsix, gt_id_user_rows_in_eledecsev, gt_id_user_rows_in_eledeceig, gt_id_user_rows_in_eledecnin, gt_id_user_rows_in_eledecten, gt_id_user_rows_in_eledecele, gt_id_user_rows_in_eledectwe)
