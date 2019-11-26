import pandas as pd
import numpy as np
import random
from datetime import datetime
from utils import *


def addHours(filename):

	df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
	size = len(df["id_user"])
	

	listDfn = []
	
	
	for i in range(size):
		if i % 1000 == 0:
			print("i = ",i,end='\r')
		listDfn.append([df["id_user"][i],df["date"][i],"DEL",df["id_item"][i],df["price"][i],df["qty"][i]])

	dfn = np.asarray(listDfn)
	dfn = pd.DataFrame(data=np.asarray(dfn))
	dfn.columns = ["id_user", "date", "hours", "id_item","price","qty"]
	dfn.to_csv('out/stage6.csv', encoding='utf-8',index=False)

addHours('out/stage5.csv')

