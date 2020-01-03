
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def	plot_db(Xgt, Xdt, encoder_model):
	np.random.shuffle(Xgt)
	np.random.shuffle(Xdt)

	Xgt = Xgt[0: 5000]
	Xgt_encoded = encoder_model.predict(Xgt)

	Xdt = Xdt[0: 5000]
	Xdt_encoded = encoder_model.predict(Xdt)

	print(Xgt[0:3])
	print(Xgt_encoded[0:3])

	fig = plt.figure()
	ax = plt.axes(projection='3d')

	# Data for three-dimensional scattered points
	#Xgt
	#xdata, ydata, zdata = Xgt[:, 0], Xgt[:, 1], Xgt[:, 2]
	#ax.scatter3D(xdata, ydata, zdata, c="r");

	#Xdt
	#xdata, ydata, zdata = Xdt[:, 0], Xdt[:, 1], Xdt[:, 2]
	#ax.scatter3D(xdata, ydata, zdata, c="b");

	#Xgt encoded
	xdata_encoded, ydata_encoded, zdata_encoded = Xgt_encoded[:, 0], Xgt_encoded[:, 1], Xgt_encoded[:, 2]
	ax.scatter3D(xdata_encoded, ydata_encoded, zdata_encoded, c="r");

	#Xdt encoded
	xdata_encoded, ydata_encoded, zdata_encoded = Xdt_encoded[:, 0], Xdt_encoded[:, 1], Xdt_encoded[:, 2]
	ax.scatter3D(xdata_encoded, ydata_encoded, zdata_encoded, c="b");

	plt.show()
