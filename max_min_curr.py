import pandas as pd
import matplotlib.pyplot as plt
import cvr
import stocks
import pdf


def frm(s):
    s = str(s)
    while len(s) < 8:
        s += " "
    return s


pr = pdf.PDF("pdf/savex.pdf")
coverWidth = 90
while True:
    c = cvr.Cover(stocks.stock["Close Price"], coverWidth)
    x = list()
    y = list()
    while c.move_forward():
        # print(frm(c.min) + " " + frm(c.max) + " " + c.get_cover())
        x.append((c.curr - c.min) / (c.max - c.min))
        y.append(c.delta)
    plt.title(stocks.get_name().split(".")[0].split("/")[1])
    plt.xlabel("(curr-min)/(max-min) | max,min in past " + str(coverWidth) + " days")
    plt.ylabel("curr-price_on_previous_day")
    plt.scatter(x, y, s=2)
    pr.add()
    plt.clf()
    if not stocks.next_stock():
        break

pr.save()
