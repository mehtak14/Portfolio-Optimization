import stocks
import math
import numpy as np
import stock_iterator
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdf
import pandas as pd
import math
import mylib

matplotlib.rcParams["lines.linewidth"] = 0.5
train_data_percent = 0.9
# Load the data, and make training and testing sets.
pdf1 = pdf.PDF("pdf/MKV3.pdf")
xx = pd.Series(stocks.get_by_name("TCS"))
data_ary = np.array(pd.Series(stocks.get_by_name("TCS")).diff().dropna())
for i in range(len(data_ary)):
    data_ary[i] = data_ary[i] / xx.iloc[i] * 100
data = pd.Series(data_ary)
train_test_divider_index = int(train_data_percent * len(data))
train_data = data[:train_test_divider_index]
test_data = data[train_test_divider_index:]
train_length = len(train_data)
test_length = len(test_data)
plt.plot(train_data)
plt.plot(test_data)
pdf1.add()
plt.clf()

# Get brackets

brk = pd.Series(0, dtype=float, index=range(-10, 11))
for x in train_data:
    idx = int(x + 0.5)
    if 10 >= idx >= -10:
        brk[idx] += 1
for i in brk.index:
    brk[i] /= train_length
plt.xticks(range(-10,11))
plt.plot(brk.index, brk)
pdf1.add()
plt.clf()

# Make cdf
cbrk = pd.Series(0, dtype=float, index=range(-10, 11))
sm = 0
for i in brk.index:
    sm += brk[i]
    cbrk[i] = sm
plt.xticks(range(-10,11))
plt.plot(cbrk.index, cbrk)
pdf1.add()
plt.clf()

# make pred
train_data_per = pd.Series(dtype=float, index=train_data.index)
test_data_per = pd.Series(dtype=float, index=test_data.index)
for i in train_data_per.index:
    if -10 <= train_data[i] <= 10:
        train_data_per[i] = train_data[i]
for i in test_data_per.index:
    if -10 <= test_data[i] <= 10:
        test_data_per[i] = test_data[i]
plt.plot(train_data_per)
plt.plot(test_data_per)

pred = pd.Series(dtype=float, index=test_data.index)
psm = train_data[len(train_data) - 1]
for i in test_data.index:
    rd = np.random.random()
    bid = cbrk.searchsorted(value=rd,side='left')-10
    psm = bid
    pred[i]=psm
plt.plot(pred)
pdf1.add(show=True)
plt.clf()

pdf1.save()
