import numpy as np
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
df = pd.read_csv("../ground_truth_small.csv", sep=",")
X = np.asarray(df)
#print("col id_item", list(X[:,3]))

print("Transforming id_item")
item_ids_set = set(X[:,3])
trans_table_item = dict()
for new_id, id in enumerate(item_ids_set):
	trans_table_item[id] = new_id
for i in range(0, len(df["id_item"])):
	#print(df["id_item"][i])
	X[i][3]=trans_table_item[X[i][3]]
trans_table_item_rev = {v: k for k, v in trans_table_item.items()}

print("Transforming date")
date_set = set(X[:,1])
trans_table_date = dict()
for new_id, id in enumerate(date_set):
	trans_table_date[id] = new_id
for i in range(0, len(df["date"])):
	X[i][1]=trans_table_date[X[i][1]]
trans_table_date_rev = {v: k for k, v in trans_table_date.items()}

print("Transforming hours")
hours_set = set(X[:,2])
trans_table_hours = dict()
for new_id, id in enumerate(hours_set):
	trans_table_hours[id] = new_id
for i in range(0, len(df["hours"])):
	X[i][2]=trans_table_hours[X[i][2]]
trans_table_hours_rev = {v: k for k, v in trans_table_hours.items()}

sscaler = StandardScaler()
X = sscaler.fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=7).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))

data_clustered = []
for i in range(min(labels), max(labels)):
	data = []
	for j, label in enumerate(labels):
		if label == i:
			init_vector = list(sscaler.inverse_transform(X[j]))
			init_vector[1] = trans_table_date_rev[int(init_vector[1])]
			init_vector[2] = trans_table_hours_rev[int(init_vector[2])]
			init_vector[3] = trans_table_item_rev[int(init_vector[3])]
			data.append(init_vector)
	data_clustered.append(data)
	df = pd.DataFrame(data= np.asarray(data))
	csv = df.to_csv(index=False)
	f = open("output_"+str(i)+".csv", "w")
	f.write(csv)
	f.close()
print("Done !")
#data_clustered = np.asarray(data_clustered)
#print(type(data_clustered))



# #############################################################################
# Plot result
#import matplotlib.pyplot as plt
#
## Black removed and is used for noise instead.
#unique_labels = set(labels)
#colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
#for k, col in zip(unique_labels, colors):
#    if k == -1:
#        # Black used for noise.
#        col = [0, 0, 0, 1]
#
#    class_member_mask = (labels == k)
#
#    xy = X[class_member_mask & core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=14)
#
#    xy = X[class_member_mask & ~core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#             markeredgecolor='k', markersize=6)
#
#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()
