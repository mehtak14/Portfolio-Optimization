import os
import pandas as pd

index = 0
files = os.listdir("stkdata/")
filesx = files.copy()
for i in range(len(files)):
    files[i] = "stkdata/" + files[i]

stock = pd.read_csv(files[index])
closingPrices = stock["Close Price"].astype('float64')
print("Current Stock - "+files[index])

def next_stock():
    global index, files, stock, closingPrices
    if index < len(files) - 1:
        index += 1
        stock = pd.read_csv(files[index])
        closingPrices = stock["Close Price"]
        print("Current Stock - "+files[index])
        return True
    else:
        return False

def get_by_name(n):
    for x in filesx:
        if(n+".csv"==x):
            return pd.read_csv("stkdata/"+n+".csv").dropna()["Close Price"]
    return None
def get_name():
    return files[index].split(".")[0]


def get_only_name():
    return get_name().split(".")[0].split("/")[1]

