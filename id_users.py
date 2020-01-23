import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import hashlib

from tools import *

def generate_alone_users(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.str,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["date"])
    date_dict = {}
    final_dict= {}
    dfn = np.asarray(df)
    for i in range(0,size):
        if df["date"][i] not in date_dict:
            date_dict[str(df["date"][i])] = []
        date_dict[str(df["date"][i])].append(dfn[i])

    save_filename = ".alone_users_new/out"
    nb_file = 0

    for date in date_dict:
        out_file = open(save_filename + str(nb_file) +".csv","w")
        for i,id_item in enumerate(date_dict[str(date)]):
            compare = [date_dict[str(date)][i][0] for i in range(0, len(date_dict[str(date)]))]
            if compare.count(id_item[0]) <= 5:
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

def delete_alone_users(stage_prec,alone):
    gt = open(stage_prec,"r")
    gt_list = []

    #ON STOCKE LES FICHIERS EN MEMEOIRE
    for line in gt:
        gt_list.append(line.split(","))

    alones_files = list()
    alones_files_list = list()

    for nb_file in range(0,37):
        alones_files.append(open(alone + "/out"+str(nb_file)+".csv","r"))

    for index,file in enumerate(alones_files):
        alones_files_list.append(list())
        for line in file:
            alones_files_list[index].append(line.split(","))

    # DELETE ALONE ITEM FROM GROUND TRUTH
    for alone_file in alones_files_list:
        gt_list = delete_alone_list_users(gt_list,alone_file)

    convert_list_to_CSV("out_new/stage7.csv",gt_list)
