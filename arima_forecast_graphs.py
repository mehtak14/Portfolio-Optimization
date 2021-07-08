import stocks
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import pdf
import pandas as pd

pr = pdf.PDF("pdf/arima_input.pdf")


def input_pdq():
    x = tuple(input(stocks.get_only_name() + " : Enter, p d q : ").strip().split(" "))
    if len(x) < 3:
        x = (1, 1, 1)
    y = [int(i) for i in x]
    return y


def arima_man_forecast(ts, f, order=None):
    confirm = False
    if order is None:
        confirm = True
    train = ts[:int(len(ts) * f)]
    test = ts[int(len(ts) * f):]
    if order is None:
        order = input_pdq()
    model = ARIMA(train, order)
    model_fit = model.fit(disp=-1)
    plt.plot(test)
    plt.plot(train)

    an = pd.Series(model_fit.forecast(len(ts) - int(len(ts) * f), alpha=0.05)[0], index=test.index)
    plt.plot(an)
    model_fit.plot_predict(start=int(len(ts) * f), end=int(len(ts) * 1.2), dynamic=False, ax=plt.gca())
    plt.title(stocks.get_name().split(".")[0].split("/")[1] + "[p,d,q]:" + str(order))
    if confirm:
        pr.add(add=False, show=True)
        y = input("Enter to confirm : ")
        if y == "":
            arima_man_forecast(ts, f, order)
        else:
            arima_man_forecast(ts, f)
    else:
        pr.add()
        plt.clf()


while True:
    arima_man_forecast(stocks.stock["Close Price"], 0.6)
    if not stocks.next_stock():
        break

pr.save()
