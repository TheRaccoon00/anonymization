import pandas as pd
import numpy as np
import random
from datetime import datetime
from utils import *


def k_anonymisation(filename):
	
	# 1 day = 86400000
	usersWeight = 10 * 86400000
	dateWeight = 10 * 86400000
	itemsWeight = 10 * 86400000
	priceWeight = 10 * 86400000
	qtyWeight = 10 * 86400000

	firstDate = 1291158000000

	df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
	size = len(df["id_user"])
	matchedRows = []
	dfnList = []
	for i in range(size - 1):
		print("i = ",i)
		distList = []
		if i in matchedRows:
			continue
		else:
			matchedRows.append(i)
			for j in range(i+1,size):
				print("j = ",j,end='\r')

				if j not in matchedRows:
					distUser = 0
					distItem = 0
					totalDist = 0
					if df["id_user"][i] != df["id_user"][j]:
						distUser += 1
					if df["id_item"][i] != df["id_item"][j]:
						distItem += 1
					distUser = distUser * usersWeight
					distDate = abs(int(df["date"][i]) - int(df["date"][j])) * dateWeight
					distItem = distItem * itemsWeight
					distPrice = abs(df["price"][i] - df["price"][j]) * priceWeight						
					distQty = abs(df["qty"][i] - df["qty"][j]) * qtyWeight
					totalDist = ((distUser)**2 + (distDate)**2 + (distItem)**2 + (distPrice)**2 + (distQty)**2)**0.5
					distList.append([totalDist,j])
					sortedDistList = sortedByColumn(distList,0)
					matchedRows.append(sortedDistList[0][1])
					vector1 = [df["id_user"][i]] + [df["date"][i]] + [df["id_item"][i]] + [df["price"][i]] + [df["qty"][i]]
					vector2 = [df["id_user"][sortedDistList[0][1]]] + [df["date"][sortedDistList[0][1]]] + [df["id_item"][sortedDistList[0][1]]] + [df["price"][sortedDistList[0][1]]] + [df["qty"][sortedDistList[0][1]]]
					newVector = []
					alea = random.randrange(0,2)
					if alea == 0:
						newVector.append(df["id_user"][i])
						newVector.append(df["date"][sortedDistList[0][1]])
						newVector.append(df["id_item"][i])
						newVector.append(df["price"][sortedDistList[0][1]])
						newVector.append(df["qty"][i])
					else:
						newVector.append(df["id_user"][sortedDistList[0][1]])
						newVector.append(df["date"][i])
						newVector.append(df["id_item"][sortedDistList[0][1]])
						newVector.append(df["price"][i])
						newVector.append(df["qty"][sortedDistList[0][1]])
					dfnList.append(newVector)
					dfnList.append(newVector) # it's normal


	"""
	sortedDfnList = sortedByColumn(dfnList,1)

	print(sortedDfnList)

	for i in range(len(sortedDistList)):
		date = datetime.fromtimestamp((int(sortedDfnList[i][1]) + 1291158000000)/1000.0)
		year = str(date.year)
		month = str(date.month)
		if date.day > 9:
			day = str(date.day)
		else:
			day = "0" + str(date.day)
		sortedDfnList[i][1] = year + "/" + month + "/" + day
		"""
	print("Be brave")
	dfn = np.asarray(dfnList)
	df = pd.DataFrame(data=np.asarray(dfn))
	df.columns = ["id_user", "date", "id_item","price","qty"]
	df.to_csv('out/stage5.csv', encoding='utf-8',index=False)
	print("END")


"""
k_anonymisation('out/stage4.csv')
"""

