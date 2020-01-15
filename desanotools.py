import pandas as pd
import time, random, sys, os, argparse, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def compact_conf(use_index, use_dates, use_hours, use_items, use_scaler,\
	force_year_equality, force_month_equality, force_day_equality,\
	force_item_equality, force_qtt_equality, nb_threads, show_result,\
	id_index, date_index, hours_index, item_index, price_index,\
	qtt_index, gt_path, dt_path, out_path):
	conf = {
		"use_index": use_index,
		"use_dates": use_dates,
		"use_hours": use_hours,
		"use_items": use_items,
		"use_scaler": use_scaler,
		"force_year_equality": force_year_equality,
		"force_month_equality": force_month_equality,
		"force_day_equality": force_day_equality,
		"force_item_equality": force_item_equality,
		"force_qtt_equality": force_qtt_equality,
		"nb_threads": nb_threads,
		"show_result": show_result,
		"id_index": id_index,
		"date_index": date_index,
		"hours_index": hours_index,
		"item_index": item_index,
		"price_index": price_index,
		"qtt_index": qtt_index,
		"gt_path": gt_path,
		"dt_path": dt_path,
		"out_path": out_path
	}
	return conf

def uncompact_conf(conf):
	return conf["use_index"], conf["use_dates"], conf["use_hours"],\
	conf["use_items"], conf["use_scaler"], conf["force_year_equality"],\
	conf["force_month_equality"], conf["force_day_equality"], conf["force_item_equality"],\
	conf["force_qtt_equality"], conf["nb_threads"], conf["show_result"],\
	conf["id_index"], conf["date_index"], conf["hours_index"], conf["item_index"],\
	conf["price_index"], conf["qtt_index"], conf["gt_path"],\
	conf["dt_path"], conf["out_path"]

def save_conf(conf, file_path):
	f = open(file_path, "w")
	f.write(json.dumps(conf))
	f.close()

def load_conf(file_path):
	f = open(file_path, "r")
	raw = f.read()
	conf = json.loads(raw)
	return conf

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

def learn_panda_to_vector(df, scaler, conf):
	use_index, use_dates, use_hours, use_items, use_scaler, force_year_equality,\
	force_month_equality, force_day_equality, force_item_equality,\
	force_qtt_equality, nb_threads, show_result, id_index, date_index,\
	hours_index, item_index, price_index, qtt_index,\
	gt_path, dt_path, out_path = uncompact_conf(conf)

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

def get_similar(gt, Xgt, Xdt_row, conf, encoder_model, return_length=10):

	use_index, use_dates, use_hours, use_items, use_scaler, force_year_equality,\
	force_month_equality, force_day_equality, force_item_equality,\
	force_qtt_equality, nb_threads, show_result, id_index, date_index,\
	hours_index, item_index, price_index, qtt_index,\
	gt_path, dt_path, out_path = uncompact_conf(conf)

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
		Xdt_row = np.asarray([Xdt_row])
		#print(Xgt[0])
		#print(Xdt_row[0])
		Xgt_encoded = encoder_model.predict(Xgt)
		Xdt_row_encoded = encoder_model.predict(Xdt_row)
		eds = euclidean_distances(Xgt_encoded, Xdt_row_encoded)
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

		#gt_res = list(gt.loc[distances[i][0]])
		#[print(list(gt.loc[distances[i][0]]), distances[i], init_Xgt[distances[i][0]]) for i in range(0, len(distances))]
		#exit()
		for i in range(0, min(len(distances), return_length)):
			#(Xgt index, Xgt_vect)
			#euclidian distance / score
			similar_rows.append((distances[i][0], init_Xgt[distances[i][0]]))
			similar_rows_score.append(distances[i][1])

		del distances, eds, stacked

	del Xgt, Xgt_indexes

	return np.asarray(similar_rows), similar_rows_score

def data_to_vector(conf, data, scaler, trans_table_date_y, trans_table_date_m, trans_table_date_d, trans_table_hours, trans_table_item):
	use_index, use_dates, use_hours, use_items, use_scaler, force_year_equality,\
	force_month_equality, force_day_equality, force_item_equality,\
	force_qtt_equality, nb_threads, show_result, id_index, date_index,\
	hours_index, item_index, price_index, qtt_index,\
	gt_path, dt_path, out_path = uncompact_conf(conf)

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

def reverse_vector(vec, scaler, conf, trans_table_date_y_rev, trans_table_date_m_rev, trans_table_date_d_rev, trans_table_hours_rev, trans_table_item_rev):
	use_index, use_dates, use_hours, use_items, use_scaler, force_year_equality,\
	force_month_equality, force_day_equality, force_item_equality,\
	force_qtt_equality, nb_threads, show_result, id_index, date_index,\
	hours_index, item_index, price_index, qtt_index,\
	gt_path, dt_path, out_path = uncompact_conf(conf)

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

def export_f_file(gtn, sfn, dfn, out_path):
	all_gt_id_user = list(set(gtn[:,0].astype(str).tolist()))
	id_user_set = sorted(list(set(dfn[:,0].astype(str).tolist())))

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

	os.remove()
	f_file = open(out_path, "a")
	f_file.write("id_user,0,1,2,3,4,5,6,7,8,9,10,11,12\n")

	for vec in res:
		f_file.write(','.join(vec)+"\n")
	f_file.close()

def	find_shopping_list(dtn, id_index):
	#return items for each user with it's dt index
	#ids = [str(dtn[i][id_index]) for i in range(0, dtn.shape[0])]
	shopping_lists = dict().gtn
	for index, dtn_row in enumerate(dtn):
		if shopping_lists.get(str(dtn_row[id_index]), None) == None:
			shopping_lists[str(dtn_row[id_index])] = list()

		shopping_lists[str(dtn_row[id_index])].append((index, dtn_row))
	#

	print(shopping_lists)
	return shopping_lists


def output(gt, result, out_path):
	gtn = np.asarray(gt)
	result = {v: k for k, v in result.items()}

	id_users_gt = gtn[:,0]
	print(id_users_gt)
	id_users_dt = list(result.values())
	print(id_users_dt)
	missing_ids = list(set(id_users_gt)-set(list(result.keys())))
	print(missing_ids)

	out_dict = dict()

	for gtn_row in gtn:
		id_user = gtn_row[0]
		date = split_date(np.asarray([gtn_row]), 1)
		y, m, d = date[0][1], date[0][2], date[0][3]
		print(type(y), m, d)
		if out_dict.get(id_user, None) == None:
			out_dict[id_user] = ["DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL"]
		print(out_dict[id_user])
		if y == "2010":
			out_dict[id_user][0] = result.get(id_user, "DEL")
		elif y == "2011":
			if m == "1":
				out_dict[id_user][1] = result.get(id_user, "DEL")
			if m == "2":
				out_dict[id_user][2] = result.get(id_user, "DEL")
			if m == "3":
				out_dict[id_user][3] = result.get(id_user, "DEL")
			if m == "4":
				out_dict[id_user][4] = result.get(id_user, "DEL")
			if m == "5":
				out_dict[id_user][5] = result.get(id_user, "DEL")
			if m == "6":
				out_dict[id_user][6] = result.get(id_user, "DEL")
			if m == "7":
				out_dict[id_user][7] = result.get(id_user, "DEL")
			if m == "8":
				out_dict[id_user][8] = result.get(id_user, "DEL")
			if m == "9":
				out_dict[id_user][9] = result.get(id_user, "DEL")
			if m == "10":
				out_dict[id_user][10] = result.get(id_user, "DEL")
			if m == "11":
				out_dict[id_user][11] = result.get(id_user, "DEL")
			if m == "12":
				out_dict[id_user][12] = result.get(id_user, "DEL")
		print(out_dict[id_user])

	for ids in missing_ids:
		out_dict[ids] = ["DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL"]

	out_list = np.asarray([[key]+out_dict[key] for key in list(out_dict.keys())])

	out_list_sorted = out_list[out_list[:, 0].argsort()]
	print(out_list_sorted[0:3])
	exit()

	#for missing_id_user in missing_ids:
	#	vec = [str(missing_id_user), "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL", "DEL"]
	#	res.append(vec)

	#todo : write in file

	res = sorted(res, key = lambda x: x[0])

	if os.path.exists(out_path):
		os.remove(out_path)

	f_file = open(out_path, "a")
	f_file.write("id_user,0,1,2,3,4,5,6,7,8,9,10,11,12\n")

	for vec in out_list_sorted:
		f_file.write(','.join(vec)+"\n")
	f_file.close()

	#for key in list(out_dict.keys()):
		#pass

if __name__ == '__main__':
	gt = np.asarray([["AZERTY", "2010/12/01", 3, 8, 9], ["ZERTYU", "2011/01/01", 4, 9, 10], ["ERTYUI", "2011/02/01", 4, 9, 10]])
	result = {"YTVYGJH":"AZERTY", "UJNKLZ": "ZERTYU"}
	out_path = "F_File.pute"
	output(gt, result, out_path)
