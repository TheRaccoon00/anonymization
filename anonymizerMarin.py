import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
from pandas import DataFrame, Series
import hashlib
from utils import *

def main():
    #main
    print(" --- START --- ")
    #pseudo('ground_truth.csv')
    #del_hours('out/stage1.csv')
    #megaMixUp('out/stage2.csv')
    vectorizer('out/stage2.csv')
    k_anonymisation2('out/stage4.csv')
    addHours('out/stage5.csv')
    print(" --- END --- ")
    #main

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
    f = open("/home/clement/Documents/4a/projetSecurite/sel.txt","r")
    sel = f.read()
    for i in range(0,len(df["id_user"])):
        dfn[i][0] = int(hashlib.sha512((str(trans_table[df["id_user"][i]])+sel).encode()).hexdigest(),16)
    #Create new CSV
    df = pd.DataFrame(data=np.asarray(dfn))
    print(" --- STAGE 1 DONE --- ")
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('out/stage1.csv', encoding='utf-8',index=False)

def del_hours(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    for col in df.columns:
        df['hours'].values[:] = 'DEL'
    print(" --- STAGE 2 DONE --- ")
    df.to_csv('out/stage2.csv',encoding='utf-8',index=False)

def megaMixUp(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["id_user"])
    data = []
    for i in range(size):
        print("i = ",i,end='\r')
        l = [df["id_user"][i]] + [df["date"][i]] + [df["id_item"][i]] + [df["price"][i]] + [df["qty"][i]] + [random.randrange(0,1000000000000000000)]
        data.append(l)
    print("DONE")
    sortedData = sortedByColumn2(data,5)
    sortedData = np.asarray(sortedData)
    newData = sortedData[:,:-1]
    dfn = np.asarray(newData)
    df = pd.DataFrame(data=np.asarray(dfn))
    print(" --- STAGE 3 DONE --- ")
    df.columns = ["id_user", "date", "id_item","price","qty"]
    df.to_csv('out/stage3.csv', encoding='utf-8',index=False)

def vectorizer(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["id_user"])
    firstDate = 1291158000000
    vectorSet = []
    for i in range(size):
        print("i = ",i,end='\r')
        l = []
        dt = datetime(int(df["date"][i][:4]),int(df["date"][i][5:7]),int(df["date"][i][8:]))
        vectorizedDate = int(round(dt.timestamp() * 1000)) - firstDate
        l = [df["id_user"][i]] + [df["date"][i]]  + [df["id_item"][i]] + [df["price"][i]] + [df["qty"][i]] + [vectorizedDate]
        vectorSet.append(l)
    dfn = np.asarray(vectorSet)
    df = pd.DataFrame(data=np.asarray(dfn))
    print(" --- STAGE 4 DONE --- ")
    df.columns = ["id_user", "date", "id_item","price","qty","msDate"]
    df.to_csv('out/stage4.csv', encoding='utf-8',index=False)

def k_anonymisation(filename):
    
    # 1 day = 86400000
    usersWeight = 10 * 86400000
    dateWeight = 10 * 86400000
    itemsWeight = 10 * 86400000
    priceWeight = 10 * 86400000
    qtyWeight = 10 * 86400000

    firstDate = 1291158000000

    print("START")

    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"id_item":np.object,"price":np.float,"qty":np.int,"msDate":np.int})
    size = len(df["id_user"])
    matchedRows = []
    dfnList = []
    for i in range(size - 1):
        if i % 1000 == 0:
            print("i = ",i,end='\r')
        distList = []
        if i in matchedRows:
            continue
        else:
            matchedRows.append(i)
        k = min(i + 1 + 2,size)
        n = 0
        for j in range(i+1,k):
            if j in matchedRows:
                continue

            else:
                distUser = 0
                distItem = 0
                totalDist = 0
                if df["id_user"][i] != df["id_user"][j]:
                    distUser += 1
                if df["id_item"][i] != df["id_item"][j]:
                    distItem += 1
                distUser = distUser * usersWeight
                distDate = abs(int(df["msDate"][i]) - int(df["msDate"][j])) * dateWeight
                distItem = distItem * itemsWeight
                distPrice = abs(df["price"][i] - df["price"][j]) * priceWeight                      
                distQty = abs(df["qty"][i] - df["qty"][j]) * qtyWeight
                totalDist = ((distUser)**2 + (distDate)**2 + (distItem)**2 + (distPrice)**2 + (distQty)**2)**0.5
                distList.append([totalDist,j])
        sortedDistList = sortedByColumn1(distList,0)
        matchedRows.append(sortedDistList[0][1])
        vector1 = [df["id_user"][i]] + [df["msDate"][i]] + [df["id_item"][i]] + [df["price"][i]] + [df["qty"][i]]
        vector2 = [df["id_user"][sortedDistList[0][1]]] + [df["msDate"][sortedDistList[0][1]]] + [df["id_item"][sortedDistList[0][1]]] + [df["price"][sortedDistList[0][1]]] + [df["qty"][sortedDistList[0][1]]]
        newVector1 = []
        newVector2 = []
        newVector1.append(df["id_user"][i])
        newVector1.append(df["date"][sortedDistList[0][1]])
        newVector1.append(df["id_item"][sortedDistList[0][1]])
        newVector1.append(df["price"][i])
        newVector1.append(df["qty"][i])
        newVector1.append(df["msDate"][sortedDistList[0][1]])
        
        newVector2.append(df["id_user"][sortedDistList[0][1]])
        newVector2.append(df["date"][i])
        newVector2.append(df["id_item"][i])
        newVector2.append(df["price"][sortedDistList[0][1]])
        newVector2.append(df["qty"][sortedDistList[0][1]])
        newVector2.append(df["msDate"][i])
        dfnList.append(newVector1)
        dfnList.append(newVector2) # it's normal


    sortedDfnList = sortedByColumn2(dfnList,5)

    print(sortedDfnList)

    newDfn = np.array(sortedDfnList)
    newDfn = newDfn[:,:-1]

    print("START WRITE CSV")
    dfn = np.asarray(newDfn)
    df2 = pd.DataFrame(data=np.asarray(dfn))
    df2.columns = ["id_user", "date", "id_item","price","qty"]
    df2.to_csv('out/stage5.csv', encoding='utf-8',index=False)
    print("END")

def k_anonymisation2(filename):
    
    # 1 day = 86400000
    usersWeight = 10 * 86400000
    dateWeight = 10 * 86400000
    itemsWeight = 10 * 86400000
    priceWeight = 10 * 86400000
    qtyWeight = 10 * 86400000

    firstDate = 1291158000000

    print("START")

    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"id_item":np.object,"price":np.float,"qty":np.int,"msDate":np.int})
    size = len(df["id_user"])
    matchedRows = []
    dfnList = []
    for i in range(1,int(size/2)+1):
        if i % 1000 == 0:
            print("i = ",i,end='\r')

        newVector1 = []
        newVector2 = []
        newVector1.append(df["id_user"][i])
        newVector1.append(df["date"][size-i])
        newVector1.append(df["id_item"][size-i])
        newVector1.append(df["price"][i])
        newVector1.append(df["qty"][i])
        newVector1.append(df["msDate"][size-i])
        
        newVector2.append(df["id_user"][size-i])
        newVector2.append(df["date"][i])
        newVector2.append(df["id_item"][i])
        newVector2.append(df["price"][size-i])
        newVector2.append(df["qty"][size-i])
        newVector2.append(df["msDate"][i])
        dfnList.append(newVector1)
        dfnList.append(newVector2) # it's normal


    sortedDfnList = sortedByColumn2(dfnList,5)

    newDfn = np.array(sortedDfnList)
    newDfn = newDfn[:,:-1]

    print("START WRITE CSV")
    dfn = np.asarray(newDfn)
    df2 = pd.DataFrame(data=np.asarray(dfn))
    df2.columns = ["id_user", "date", "id_item","price","qty"]
    df2.to_csv('out/stage5.csv', encoding='utf-8',index=False)
    print("END")


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





if __name__ == "__main__":
    main()