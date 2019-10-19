"""
File: attack.py
Author: Cat1 Narvali
Email: cat1narvali@gmail.com
Github: https://github.com/TheRaccoon00/anonymization
Description: Database anonymizer for DARC competition
"""

import pandas as pd
import time

def get_nb_of_id(user_id, table):
	return sum([1 for m in table["id_user"] == user_id if m == True])

def avg_price(table):
	return sum(table["price"])/table["price"].shape[0]

def ca_price(table):
	price = table["price"]
	qty = table["qty"]
	ca = 0
	for i in range(0, price.shape[0]):
		ca = ca + price[i]*qty[i]
	return ca

def nb_bought_item(table, id_item):
	return sum([1 for m in table["id_item"] == id_item if m == True])


def main():
	gt = pd.read_csv("ground_truth.csv", sep=",")
	dt = pd.read_csv("/home/theoguidoux/INSA/ws/projetsecu4a/docs/CSV_RENDU/S_Godille_Table_1.csv", sep=",")

#	print("############## avg_price ##############")
#	print("gt :", avg_price(gt))
#	print("dt :", avg_price(dt))
#
#	print("############## ca_price ##############")
#	print("gt :", ca_price(gt))
#	print("dt :", ca_price(dt))

	print("############## nb_bought_item ##############")
	print("gt :", nb_bought_item(gt,"21080"))
	print("dt :", nb_bought_item(dt,"21080"))


if __name__ == "__main__":
    main()
