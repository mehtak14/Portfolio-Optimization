import stocks
import math
import numpy as np
import stock_iterator
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdf
import matplotlib
matplotlib.rcParams["lines.linewidth"] = 0.5

def mav_ary(l, x):
    a = np.empty(l.size)
    s = 0
    mav = np.mean(l[0:x])
    a[:x] = mav
    for e in range(x, l.size):
        a[e] = mav + (l[e] - l[s]) / x

        mav = a[e]
        s += 1
    return a


def mn_ary(l, x):
    b = np.empty(l.size)
    s = 0
    mn = np.min(l[0:x])
    b[:x] = mn
    for e in range(x, l.size):
        if mn == l[s]:
            b[e] = np.min(l[s + 1:e + 1])
        elif l[e] < mn:
            b[e] = l[e]
        else:
            b[e] = mn
        mn = b[e]
        s += 1

    return b


def dtr(l, x):
    b = mn_ary(l, x)

    plt.plot(l)
    plt.plot(mav_ary(b, x))
    mav_mn = mav_ary(b, x)
    mx = np.max(mav_mn)
    dt = np.empty(l.size)
    for i in range(l.size):
        dt[i] = l[i] / ((1 + mav_mn[i] / mx) ** 3) * 3
    plt.plot(dt)
    pd.add()
    plt.clf()
    return dt


def f(l, nm):
    l = np.array(l)
    n = l.size
    w_size = 500

    plt.title(nm)
    plt.plot(l)
    pd.add()
    plt.clf()

    # Split Data
    ntrain = int(0.75 * n)
    ntest = n - ntrain
    train = l[:ntrain]
    test = l[ntrain:]
    plt.title("Split Data")
    plt.plot(range(ntrain), train)
    plt.plot(range(ntrain, n), test)
    pd.add()
    plt.clf()
    plt.title("Remove Trend")
    dt = dtr(l, w_size)

    # Fit Normal
    train = dt
    ntrain = train.size

    def ms(trn):
        dif = np.diff(trn)
        mu, sg = norm.fit(dif)
        plt.title("Fit Normal | mean = " + str(mu)[:5] + " sd = " + str(sg)[:5])
        plt.hist(dif, bins=np.arange(-70, 70, 1), density=True)
        plt.plot(np.arange(-70, 70, 1), norm.pdf(np.arange(-70, 70, 1), mu, sg))
        pd.add()
        plt.clf()
        return mu, sg

    mu1, sg1 = ms(train[0:int(ntrain / 5)])
    print(mu1, sg1)
    mu2, sg2 = ms(train[int(ntrain / 5):2 * int(ntrain / 5)])
    print(mu2, sg2)
    mu3, sg3 = ms(train[2 * int(ntrain / 5):3 * int(ntrain / 5)])
    print(mu3, sg3)
    mu4, sg4 = ms(train[3 * int(ntrain / 5):4 * int(ntrain / 5)])
    print(mu4, sg4)
    mu5, sg5 = ms(train[4 * int(ntrain / 5):])
    print(mu5, sg5)

    mu = (mu1 + mu2 + mu3 + mu4 + mu5) / 5
    sg = (sg1 + sg2 + sg3 + sg4 + sg5) / 5

    brt = (mu + sg * sg) / 2
    drt = (sg * sg - mu) / 2
    print(brt, drt)


pd = pdf.PDF("pdf/ab.pdf")
# f(stocks.get_by_name("TCS"), "TCS")
stock_iterator.iterate(f)
pd.save()