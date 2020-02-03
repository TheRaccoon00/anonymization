

import pickle
from desanotools import *

data = None
with open('F_S_1-lcoinsti_by_cat1.pickle', 'rb') as config_dictionary_file:
	data = pickle.load(config_dictionary_file)

gt = data["gt"]
dt = data["dt"]
result = data["result"]
out_path = "F_S_1-lcoinsti_by_cat1_v2.csv"

output2(dt, result, out_path)
