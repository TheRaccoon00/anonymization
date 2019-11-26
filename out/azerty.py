import pandas as pd

data = pd.read_csv("stage5.csv", sep=",")
nbRows = data.shape

print(nbRows)