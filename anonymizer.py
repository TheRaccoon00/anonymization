"""
File: anonymizer.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database anonymizer for DARC competition
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import hashlib
from stats import *

def main():
    pseudo('ground_truth.csv')
    modificateDateV2('out/stage1.csv')
    deleteRandomColumn('out/stage2.csv')

def pseudo(filename):
    #Import CSV file
    df =  pd.read_csv(filename,dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    id_user_set = set(df["id_user"])
    trans_table = dict()
    #Create new id
    for new_id, id in enumerate(id_user_set):
        trans_table[id] = new_id
    dfn = np.asarray(df)
    #Hash and store new id
    for i in range(0,len(df["id_user"])):
        dfn[i][0] = int(hashlib.sha512((str(trans_table[df["id_user"][i]])+"Il utilise plutôt un système de refroidissement par évaporation avancé qui tire les eaux grises d'un canal industriel situé à proximité.").encode()).hexdigest(),16)
        #dfn[i][0] = trans_table[df["id_user"][i]]
    #Create new CSV
    df = pd.DataFrame(data=np.asarray(dfn))
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('out/stage1.csv', encoding='utf-8',index=False)

def modificateDateV2(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
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
    df.to_csv('out/stage2.csv', encoding='utf-8',index=False)

def deleteRandomColumn(filename):
    file = open(filename,"r")
    out  = open("out/stage3.csv","w")
    number = 1

    for line in file:
        if number%12 != 0:
            out.write(str(line))
            number+=1
        else:
            out.write("DEL\n")
            number+=1

    file.close()
    out.close()

if __name__ == "__main__":
    main()
