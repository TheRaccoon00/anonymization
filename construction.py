import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import hashlib

#from stats import *
from tools import *

def pseudo(filename):
    #Import CSV file
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.str_})
    id_user_set = set(df["id_user"])
    trans_table = dict()
    #Create new id
    for new_id, id in enumerate(id_user_set):
        trans_table[id] = new_id
    dfn = np.asarray(df)
    #Hash and store new id
    for i in range(0,len(df["id_user"])):
        #dfn[i][0] = int(hashlib.sha512((str(trans_table[df["id_user"][i]])+"Il utilise plutôt un système de refroidissement par évaporation avancé qui tire les eaux grises d'un canal industriel situé à proximité.").encode()).hexdigest(),16)
        dfn[i][0] = trans_table[df["id_user"][i]]
    #Create new CSV
    df = pd.DataFrame(data=np.asarray(dfn))
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('out_new/stage8.csv', encoding='utf-8',index=False)

def del_hours(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    for col in df.columns:
        df['hours'].values[:] = 'DEL'
    df.to_csv('out_new/stage2.csv',encoding='utf-8',index=False)

def modifyDateV2(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.str_})
    size = len(df["date"])
    listDate = []
    for i in range(size):
        listDate.append(df["date"][i])
    dfn = np.asarray(df)
    for i in range(size):
        if 20 > int(listDate[i][8:]) >= 10:
            dfn[i][1] = listDate[i][:4] + "/" + listDate[i][5:7] + "/" + "10" # 1 is for date
        elif int(listDate[i][8:]) >= 20:
            dfn[i][1] = listDate[i][:4] + "/" + listDate[i][5:7] + "/" + "20" # 1 is for date
        else:
            dfn[i][1] = listDate[i][:4] + "/" + listDate[i][5:7] + "/" + "01" # 1 is for date
    df = pd.DataFrame(data=np.asarray(dfn))
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('out_new/stage3.csv', encoding='utf-8',index=False)
