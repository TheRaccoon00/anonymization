import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import hashlib

from stats import *
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

def generate_alone_items(filename):
    df =  pd.read_csv(filename,dtype={"id_user":np.str,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    size = len(df["date"])
    date_dict = {}
    final_dict= {}
    dfn = np.asarray(df)
    for i in range(0,size):
        if df["date"][i] not in date_dict:
            date_dict[str(df["date"][i])] = []
        date_dict[str(df["date"][i])].append(dfn[i])

    save_filename = ".alone_items_new/out"
    nb_file = 0

    for date in date_dict:
        out_file = open(save_filename + str(nb_file) +".csv","w")
        for i,id_item in enumerate(date_dict[str(date)]):
            compare = [date_dict[str(date)][i][3] for i in range(0, len(date_dict[str(date)]))]
            if compare.count(id_item[3]) <= 5:
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

def distrib_items_v1(stage_prec,alone):
    gt = open(stage_prec,"r")
    gt_list = []

    distribute_list = []
    distribute_list_str = []

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

    #ON PARCOURT LES FICHIERS SEULS

    for alone_file in alones_files_list:
        for alone_line in alone_file:
            check = 0
            for gt_line in gt_list:
                item_alone = alone_line[3]
                item_gt = gt_line[3]

                date_alone = alone_line[1]
                date_gt = gt_line[1]

                if item_alone == item_gt and date_alone != date_gt:
                    gt_line[5] = str(int(alone_line[5]) + int(gt_line[5])) + '\n'
                    if check == 0:check += 1
                    else: break

                elif item_alone == item_gt and date_alone == date_gt:
                    gt_line[0] = "DEL"
                    if check == 0:check += 1
                    else: break
        print("file done")

    convert_list_to_CSV("out_new/stage4.csv",gt_list)

def gt_nb_date(gt_file):
    gt = open(gt_file,"r")
    gt_list = []

    for line in gt:
        gt_list.append(line.split(","))

    date = gt_list[0][1]
    nb=0

    for gt_line in gt_list:
        if date == gt_line[1][:7]:
            nb+=1
        else:
            print(date + " : " + str(nb))
            date = gt_line[1][:7]
            nb = 0
    print(date + " : " + str(nb))

def distrib_items_v2(stage_prec,alone):
    gt = open(stage_prec,"r")
    gt_list = []

    distribute_list = []
    distribute_list_str = []

    still_waiting_list = []

    static_nb_trans_month = {
        "2010/12":20899,
        "2011/01":16575,
        "2011/02":15377,
        "2011/03":21099,
        "2011/04":18067,
        "2011/05":21533,
        "2011/06":21047,
        "2011/07":20083,
        "2011/08":20919,
        "2011/09":30782,
        "2011/10":39236,
        "2011/11":48815,
        "2011/12":12609
        }

    modul_nb_trans_month = {
        "2010/12":0,
        "2011/01":0,
        "2011/02":0,
        "2011/03":0,
        "2011/04":0,
        "2011/05":0,
        "2011/06":0,
        "2011/07":0,
        "2011/08":0,
        "2011/09":0,
        "2011/10":0,
        "2011/11":0,
        "2011/12":0
    }

    still_waiting_list = []
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
        gt_list = delete_alone_list_items(gt_list,alone_file,modul_nb_trans_month)

    convert_list_to_CSV("out_new/stage_pre_6.csv",gt_list)

    # DISTRIBUTE ALONE ITEM IN GROUND TRUTH
    for index_alone_file,alone_file in enumerate(alones_files_list):
        start = 0
        for alone_line in alone_file:
            if start == 0:
                for index_gt,gt_line in enumerate(gt_list):
                    item_alone = alone_line[3]
                    item_gt = gt_line[3]

                    date_alone = alone_line[1]
                    date_gt = gt_line[1]

                    if item_alone == item_gt and date_alone != date_gt and modul_nb_trans_month[date_gt[:7]] < 0:
                        start = 1
                        gt_list.insert(index_gt,[alone_line[0],date_gt,alone_line[2],alone_line[3],alone_line[4],alone_line[5]])
                        modul_nb_trans_month[date_gt[:7]] += 1
                        break

            elif start == 1:
                for index_gt,gt_line in enumerate(list(reversed(gt_list))):
                    item_alone = alone_line[3]
                    item_gt = gt_line[3]

                    date_alone = alone_line[1]
                    date_gt = gt_line[1]

                    if item_alone == item_gt and date_alone != date_gt and modul_nb_trans_month[date_gt[:7]] < 0:
                        start = 0
                        gt_list.insert(len(gt_list)-index_gt,[alone_line[0],date_gt,alone_line[2],alone_line[3],alone_line[4],alone_line[5]])
                        modul_nb_trans_month[date_gt[:7]] += 1
                        break
        print("##########")
        print("File " + str(index_alone_file) + " done")
        print(modul_nb_trans_month)
        print(len(gt_list))

    print("#######")
    print(modul_nb_trans_month)
    print(len(gt_list))

    for key in modul_nb_trans_month:
        for index_gt,gt_line in enumerate(gt_list):
            try:
                if gt_line[1][:7] == key:
                    index_to_add = index_gt
                    break
            except:
                continue
        for i in range(0,abs(modul_nb_trans_month[key])):
            gt_list.insert(index_to_add,["DEL"])

    convert_list_to_CSV("out_new/stage6.csv",gt_list)
