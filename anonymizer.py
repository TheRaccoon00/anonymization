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
    #modificateDateV2('out/stage1.csv')
    #del_hours('out/stage2.csv')
    distribute_with_date("out/stage3.csv")
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


def del_hours(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    for col in df.columns:
        df['hours'].values[:] = 'DEL'
    df.to_csv('out/stage3.csv',encoding='utf-8',index=False)

def generate_alone_items(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.float64,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["date"])
    date_dict = {}
    final_dict= {}
    dfn = np.asarray(df)
    for i in range(0,size):
        if df["date"][i] not in date_dict:
            date_dict[str(df["date"][i])] = []
        date_dict[str(df["date"][i])].append(dfn[i])

    save_filename = ".alone_items/out"
    nb_file = 0

    for date in date_dict:
        out_file = open(save_filename + str(nb_file) +".csv","w")
        for i,id_item in enumerate(date_dict[str(date)]):
            compare = [date_dict[str(date)][i][3] for i in range(0, len(date_dict[str(date)]))]
            if compare.count(id_item[3]) <= 1:
                if date not in final_dict:
                    final_dict[str(date)] = []
                final_dict[str(date)].append(id_item)
                for index_item, item in enumerate(id_item):
                    if index_item != 5:
                        out_file.write(str(item) + ",")
                    else:
                        out_file.write(str(item) + "\n")

        out_file.close()
        nb_file += 1
        print(str(date) + ": " + str(len(final_dict[str(date)])))

    files = list()
    files_list = list()

    alone_items = list()

    for nb_file in range(0,13):
        files.append(open(".alone_items/out"+str(nb_file)+".csv","r"))

    for index,file in enumerate(files):
        files_list.append(list())
        for line in file:
            files_list[index].append(line)

    for file in files_list:
        for nuplet in file:
            compare = []
            for f in files_list:
                for line_compare in f:
                    compare.append(line_compare.split(",")[3])
            if compare.count(str(nuplet.split(",")[3])) <= 1:
                alone_items.append(nuplet)

    print(len(alone_items))

def distribute_with_date(filename):

    file = open(filename,"r")
    file_list = []
    for line in file:
        file_list.append(line.split(","))

    files = list()
    files_list = list()

    for nb_file in range(0,37):
        files.append(open(".alone_items/out"+str(nb_file)+".csv","r"))

    for index,file in enumerate(files):
        files_list.append(list())
        for line in file:
            files_list[index].append(line)
    count = 0
    found = 0
    for file in files_list:
        for indexfile, nuplet in enumerate(file):
            found = 0
            for file_compare in files_list:
                for nup in file_compare:
                    #print(nuplet.split(","))
                    if nuplet.split(",")[3] in nup.split(","):
                        file.pop(indexfile)
                        new_nuplet = nuplet.split(",")
                        nuplet_str = ""
                        for index,element in enumerate(new_nuplet):
                            if index == 1:
                                element = file_compare[0].split(",")[1]
                            if index != 5:
                                nuplet_str = nuplet_str + element + ","
                            else:
                                nuplet_str = nuplet_str + element + "\n"
                        print(nuplet_str.strip())
                        count+=1
                        file_list.append(nuplet_str.strip())
                        found = 1
                        break
                    if found == 1:
                        break
    print(count)

if __name__ == "__main__":
    main()
