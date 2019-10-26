"""
File: clustering.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database deanonymizer using machine learning DBSCAN
"""

import numpy as np
import pandas as pd
import time
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

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
	
def main():
	# #############################################################################
	# Data transformation
	df = pd.read_csv("../ground_truth_small.csv", sep=",")
	#convert in numpy array to speed transformations
	X = np.asarray(df)
	#scaler to make data great again
	sscaler = StandardScaler()

	#print(X[0]) => [17850 '2010/12/01' '08:26' '85123A' 2.55 6]

	#apply vectorization
	print("Transforming date")
	trans_table_date, trans_table_date_rev, X = vectorize(X, 1)

	print("Transforming hours")
	trans_table_hours, trans_table_hours_rev, X = vectorize(X, 2)

	print("Transforming id_item")
	trans_table_item, trans_table_item_rev, X = vectorize(X, 3)

	#print(X[0]) => [17850 0 10 241 2.55 6]

	#apply scaler to input
	X = sscaler.fit_transform(X)
	#print(X[0]) => [ 1.19472762  0.         -0.50816473  0.69908103 -0.08305897 -0.13099265]

	#there, data are ready : vectorized and scaled

	# #############################################################################
	# Compute DBSCAN
	#AgglomerativeClustering
	#SpectralClustering
	db = DBSCAN(eps=0.3, min_samples=12).fit(X) #min_samples = k
	#db = SpectralClustering(n_clusters=12, assign_labels="discretize").fit(X)#find n_clusters (groups) to find k-anonimity
	#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	#core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	#show metrics
	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))

	# #############################################################################
	# unvectorized data to get original input data and write clusters in files
	#data_clustered is a list of list having input culsters by culsters
	data_clustered = []

	print("Transforming back vectors")
	#for each found cluster
	for i in range(min(labels), max(labels)):
		#i = cluster identificator (just like cluster id)
		data = []#list to handle each input of class i
		for j, label in enumerate(labels):
			#we have an input of class i
			if label == i:
				#first unscale data to have reversible data
				init_vector = list(sscaler.inverse_transform(X[j]))
				#vectorize back vectors using reversible trans_tables
				init_vector[1] = trans_table_date_rev[int(init_vector[1])]
				init_vector[2] = trans_table_hours_rev[int(init_vector[2])]
				init_vector[3] = trans_table_item_rev[int(init_vector[3])]
				#init_vector has now its original form
				data.append(init_vector)

		#add current culter to global clusters handler
		data_clustered.append(data)

		#convert back numpy array to DataFrame and output to csv
		df = pd.DataFrame(data= np.asarray(data))
		csv = df.to_csv(index=False)
		f = open("output_"+str(i)+".csv", "w")
		f.write(csv)
		f.close()
	print("Done !")

if __name__ == "__main__":
    main()
