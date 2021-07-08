import stocks
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import pdf
import pandas as pd

pr = pdf.PDF("pdf/auto_arima.pdf", True)


def arima_auto_forecast(ts, f):
    train = ts[:int(len(ts) * f)]
    test = ts[int(len(ts) * f):]
    plt.plot(train)
    plt.plot(test)
    model = auto_arima(train)
    prd = model.predict(n_periods=len(test))
    plt.plot(prd)
    print(prd)
    plt.show()


while True:
    arima_auto_forecast(stocks.stock["Close Price"], 0.6)
    if not stocks.next_stock():
        break

pr.save()
