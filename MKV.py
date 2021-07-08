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
pdf1 = pdf.PDF("pdf/MKV.pdf")
data = pd.Series(stocks.get_by_name("TCS"))
train_test_divider_index = int(train_data_percent * len(data))
train_data = data[:train_test_divider_index]
test_data = data[train_test_divider_index:]
train_length = len(train_data)
test_length = len(test_data)
plt.plot(train_data)
plt.plot(test_data)

# Show trend and scale on the graph
trend_polyfit = np.poly1d(np.polyfit(train_data.index, train_data, 3))
trend_series = pd.Series(trend_polyfit(data.index), index=data.index)
plt.plot(trend_series)

plt.title("TCS")
plt.legend(["train", "test", "trend"])
pdf1.add(show=True)
plt.clf()
# Calculate scale by moving variance

rolling_std_window_size = 500
rolling_std = train_data.rolling(rolling_std_window_size).std().dropna()
plt.plot(rolling_std)
trend_std = np.poly1d(np.polyfit(rolling_std.index, rolling_std, 2))
scale_series = pd.Series(trend_std(data.index), index=data.index)
plt.plot(scale_series)
plt.title(str(rolling_std_window_size) + " day rolling sd")
pdf1.add()
plt.clf()

# Calculate the stationary series
stationary_trend = train_data - trend_series[:train_test_divider_index]

stationary_scale = [None] * train_test_divider_index
scale_factor = [None] * train_test_divider_index
for i in range(train_test_divider_index):
    scale_factor[i] = 1 + 100 / scale_series.iloc[i]

for i in range(train_test_divider_index):
    stationary_scale[i] = stationary_trend.iloc[i] * scale_factor[i]

plt.plot(scale_factor)
plt.title("Scale Factor")
pdf1.add()
plt.clf()
stationary_data = pd.Series(stationary_scale)
plt.plot(stationary_trend)
plt.plot(stationary_data)
plt.legend(["Remove Trend", "Scaled"])
plt.title("Stationary Series")
pdf1.add()
plt.clf()

# Get one day change data
daily_change = []
for i in range(1,train_test_divider_index):
    daily_change.append(stationary_scale[i]-stationary_scale[i-1])

plt.hist(daily_change,bins=np.arange(-100, 100, 1),density=True)
mn1, sd1 = norm.fit(daily_change)
lda = (sd1*sd1+mn1)/2
mu = (sd1*sd1-mn1)/2
print(mn1,sd1)
plt.plot(np.arange(-100, 100, 1), norm.pdf(np.arange(-100, 100, 1), mn1, sd1))
plt.title("mu = " + str(mn1)[:5] + " | sg = " + str(sd1)[:5] + " || birthR = "+str(lda)[:5]+" | deathR = "+str(mu)[:5])
pdf1.add()
plt.clf()

# Make prediction
plt.plot(train_data)
plt.plot(test_data)
pred = np.random.normal(mn1,sd1,test_length) + stationary_scale[-1]
pred_series = pd.Series(pred,index=test_data.index)

# Extract data from pred_series
pred_final = pd.Series(pred,index=test_data.index)
for i in pred_series.index:
    pred_final[i] = pred_series[i] / scale_factor[i-train_test_divider_index]

pred_final = pred_final + trend_series[train_test_divider_index:]
plt.plot(pred_final)
plt.legend(["train","test","forecast"])
pdf1.add(show=True)

# Save the pdf file
pdf1.save()
