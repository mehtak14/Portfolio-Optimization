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
pdf1 = pdf.PDF("pdf/MKV2.pdf")
xx = pd.Series(stocks.get_by_name("TCS"))
data_ary = np.array(pd.Series(stocks.get_by_name("TCS")).diff().dropna())
# for i in range(len(data_ary)):
#     data_ary[i]=data_ary[i]/xx.iloc[i]*100
data = pd.Series(data_ary)
train_test_divider_index = int(train_data_percent * len(data))
train_data = data[:train_test_divider_index]
test_data = data[train_test_divider_index:]
train_length = len(train_data)
test_length = len(test_data)
plt.plot(train_data)
plt.plot(test_data)
pdf1.add(show=True)
plt.clf()

# Calculate cumulative number of increase and decrease
cm_inc = [0] * train_length
cm_dec = [0] * train_length
cti = 0
ctd = 0
for i in range(train_length):
    if train_data.iloc[i] >= 0:
        cti += 1
    if train_data.iloc[i] < 0:
        ctd += 1
    cm_inc[i] = cti
    cm_dec[i] = ctd

plt.plot(cm_inc)
plt.plot(cm_dec)
pdf1.add()
plt.clf()

# Make sections in test data
ld = []
mu = []
w = 40
i = w
while i <= train_length:
    ld.append((cm_inc[i-1]-cm_inc[i-w]))
    mu.append((cm_dec[i-1]-cm_dec[i-w]))
    i += w
plt.hist(ld,bins=np.arange(min(ld),max(ld),1))
plt.show()
plt.hist(mu,bins=np.arange(min(mu),max(mu),1))
plt.show()

pdf1.save()
