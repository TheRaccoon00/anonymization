import pandas as pd

data = pd.read_csv("ground_truth.csv", sep=",")

print(data["hours"])
print(data["hours"][0][3:])