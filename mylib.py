import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def list_attributes(o):
    for x in o.__dir__():
        try:
            print(str(x) + " " + str(getattr(o, x)))
        except:
            print(str(x) + " " + "err")
        print(
            "-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------x-------")


def sin_csv(to, step, name="sine.csv"):
    x = np.arange(0, to, step)
    y = np.empty(len(x))
    for i in range(len(x)):
        y[i] = math.sin(x[i])
    plt.plot(x, y)
    plt.show()
    df = pd.DataFrame({"x": x, "Close Price": y})
    df.to_csv("stkdata/" + name)


def sin_trend(slope, to, step, name="sine_trend.csv"):
    x = np.arange(0, to, step)
    y = np.empty(len(x))
    for i in range(len(x)):
        y[i] = 5 * math.sin(x[i]) + slope * x[i]
    plt.plot(x, y)
    plt.show()
    df = pd.DataFrame({"x": x, "Close Price": y})
    df.to_csv("stkdata/" + name)


def prt(y):
    i = 0
    for x in y:
        print(str(i) + " : " + str(x))
        i += 1

# sin_csv(100,0.1)
# sin_trend(0.5, 100, 0.1)
