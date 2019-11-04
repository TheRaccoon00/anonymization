"""
File: anonymizer.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database anonymizer for DARC competition
"""

import pandas as pd
import numpy as np
import random
from pandas import DataFrame, Series

def main():
    #main
    pseudo()
    #main

def pseudo():
    df =  pd.read_csv('ground_truth.csv',dtype={"id_user":np.int,"date":np.object,"hours":np.object,"id_item":np.object,"price":np.float,"qty":np.int})
    df.head()
    df.info()
    size = len(df["id_user"])
    id_user_set = set(df["id_user"])
    print(len(id_user_set))
    trans_table = dict()
    for new_id, id in enumerate(id_user_set):
        trans_table[id] = new_id
    print(df["id_user"][0], trans_table[df["id_user"][0]])
    dfn = np.asarray(df)
    for i in range(0,size):
        dfn[i][0] = trans_table[df["id_user"][i]]
    df = pd.DataFrame(data=np.asarray(dfn))
    df.columns = ["id_user", "date", "hours", "id_item","price","qty"]
    df.to_csv('submission.csv', encoding='utf-8',index=False)
    print(df["id_user"])

if __name__ == "__main__":
    main()
