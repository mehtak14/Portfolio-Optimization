from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdf
import math
import scipy
import stock_gui as sg
import stock_iterator


def arima_insample_cumsum_fit_forecast(ts, g, order, name):
    model = ARIMA(ts, order)
    fit = model.fit(disp=-1)
    p = pd.Series(dtype=float)
    print(type(fit))
    if g == 1:
        p = fit.predict(typ="levels")
    else:
        i = int(0.1 * len(ts))
        while i < ts.index[-1]:
            p = p.append(fit.predict(start=i, end=i + g - 1, dynamic=True, typ="levels"))
            i += g
    xa = fit.fittedvalues.cumsum()
    plt.plot(xa)
    plt.plot(ts)
    plt.plot(p)
    plt.title(name + " | [p,d,q] : " + str(order) + " | gap=" + str(g))
    plt.show()


def detrn(y):
    x = scipy.signal.detrend(y)
    plt.plot(y)
    plt.plot(x)
    plt.show()

# stock_iterator.iterate(lambda ts, name: arima_insample_cumsum_fit_forecast(ts, 1, (1, 1, 1), name))
y = pd.read_csv("stkdata/AirPassengers.csv")["Close Price"]
x = scipy.signal.detrend(y)
plt.plot(y)
plt.plot(x)
plt.show()
