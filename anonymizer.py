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

def main():
    #main
    #pseudo('ground_truth.csv')
    #modificateDate('out/stage1.csv')
    #test('ground_truth.csv')
    #del_hours('out/stage2.csv')
    filter('out/stage3.csv','out/stage4.csv')
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
    for i in range(0,len(df["id_user"])):
        dfn[i][0] = int(hashlib.sha512((str(trans_table[df["id_user"][i]])+"Il utilise plutôt un système de refroidissement par évaporation avancé qui tire les eaux grises d'un canal industriel situé à proximité.").encode()).hexdigest(),16)
        #dfn[i][0] = trans_table[df["id_user"][i]]
    #Create new CSV
    df = pd.DataFrame(data=np.asarray(dfn))
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('out/stage1.csv', encoding='utf-8',index=False)

def modificateDate(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["date"])
    listDate = []
    for i in range(size):
        listDate.append(df["date"][i])

    dfn = np.asarray(df)

    for i in range(size):
        dfn[i][1] = listDate[i][:4] + "/" + listDate[i][5:7] + "/" + "01" # 1 is for date

    df = pd.DataFrame(data=np.asarray(dfn))
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('out/stage2.csv', encoding='utf-8',index=False)

def filter(filename1,filename2):
    '''Reads the second file and deletes all the lines it contains in the first file'''
    file1 = open(filename1,"r")
    file2 = open(filename2,"r")
    listfile1 = []
    listfile1 = file1.readlines()
    listfile2 = []
    listfile2 = file2.readlines()
    for i in range(len(listfile1)):
        listfile2.append(0)
    for to_delete in listfile2:
        for original in listfile1:
            if to_delete == original:
                listfile1.remove(original)
    with open('out/stage5.csv', 'w') as filteredfile:
        filteredfile.writelines("%s" % tuple for tuple in listfile1)

    file1.close()
    file2.close()


def del_hours(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    for col in df.columns:
        df['hours'].values[:] = 'DEL'
    df.to_csv('out/stage3.csv',encoding='utf-8',index=False)

if __name__ == "__main__":
    main()
