"""
File: clustering_measure.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database deanonymizer using machine learning DBSCAN
Objective : Find the k of k-anonimity database
"""

import numpy as np
import pandas as pd
import time
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

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
	#df = pd.read_csv("../ground_truth_medium.csv", sep=",")#k [10, 13]
	#S_Godille_Table_1_medium
	df = pd.read_csv("S_Godille_Table_1_medium.csv", sep=",")
	#convert in numpy array to speed transformations
	X = np.asarray(df)
	#scaler to make data great again
	#RobustScaler, StandardScaler
	sscaler = RobustScaler()

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
	ks = []
	k_clusters = []
	k_noises = []
	k_coeff = []

	for k in range(200, 225, 2):
		print("Scanning for k = "+str(k))
		db = DBSCAN(eps=0.3, min_samples=k).fit(X) #min_samples = k
		labels = db.labels_
		if(len(set(labels)) <= 1):
			break
		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)

		ks.append(k)
		k_clusters.append(n_clusters_)
		k_noises.append(n_noise_)
		k_coeff.append(abs(metrics.silhouette_score(X, labels)))

		#show metrics
		#print('Estimated number of clusters: %d' % n_clusters_)
		#print('Estimated number of noise points: %d' % n_noise_)
		#print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))

	#plt.plot(ks, k_clusters)
	#plt.plot(ks, k_noises)
	#plt.plot(ks, k_coeff)
	#plt.legend(['nb clusters', 'nb noise point', 'silhouette coefficient'])

	figure = plt.figure(figsize = (10, 10))
	plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1,right = 0.9, top = 0.9, wspace = 0, hspace = 0.1)
	axes = figure.add_subplot(3, 1, 1)
	axes.set_xlabel('k')
	axes.set_ylabel('nb clusters')
	axes.set_title('Estimated number of clusters')
	axes.scatter(ks, k_clusters, s = 50, color = 'blue')
	axes = figure.add_subplot(3, 1, 2)
	axes.set_xlabel('k')
	axes.set_ylabel('nb noise point')
	axes.set_title('Estimated number of noise points')
	axes.scatter(ks, k_noises, s = 50, color = 'red')
	axes = figure.add_subplot(3, 1, 3)
	axes.set_xlabel('k')
	axes.set_ylabel('silhouette coefficient')
	axes.set_title('Silhouette Coefficient')
	axes.scatter(ks, k_coeff, s = 50, color = 'red')

	plt.xlabel("k")

	plt.show()

	print("Done !")

if __name__ == "__main__":
    main()
