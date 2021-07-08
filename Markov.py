import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CoverM:
    def __init__(self, closingPrices, coverWidth):
        self.closingPrices = np.array(closingPrices)
        self.coverWidth = coverWidth
        self.start = 0
        self.end = coverWidth - 1
        self.curr = closingPrices[self.end]
        self.br_hi = None
        self.br_lo = None

    def init_br(self, diff):
        cs = np.cumsum(self.closingPrices)
        md = np.zeros(len(self.closingPrices))
        md[self.coverWidth - 1] = cs[self.coverWidth - 1] / self.coverWidth
        for i in range(self.coverWidth, len(md)):
            md[i] = (cs[i] - cs[i - self.coverWidth]) / self.coverWidth
        plt.plot(self.closingPrices)
        self.br_hi = md + diff
        self.br_lo = md - diff
        plt.fill_between(range(0, len(md)), self.br_lo, self.br_hi, color=(0.5, 0.5, 0.5, 0.2))
        plt.show()

    def move_forward(self):
        if self.end < len(self.closingPrices) - 1:
            self.start = self.start + 1
            self.end = self.end + 1
            self.curr = self.closingPrices[self.end]
            return True
        else:
            return False

    def get_cover(self):
        s = "[ "
        for x in range(self.start, self.end + 1):
            s += str(self.closingPrices[x]) + " "
        s += "]"
        return s

    def print_cover(self):
        print(self.get_cover())


def get_brackets(ts, diff, hl):
    ts = pd.Series(ts)
    md = ts.ewm(halflife=hl).mean()
    br_hi = md + diff
    br_lo = md - diff
    # plt.plot(ts)
    # plt.fill_between(range(0, len(md)), br_lo, br_hi, color=(0.5, 0.5, 0.5, 0.2))
    # plt.show()
    return br_lo, br_hi


def ud_array(ts, br_lo, br_hi):
    ud = []
    ts = np.array(ts)
    br_lo = np.array(br_lo)
    br_hi = np.array(br_hi)
    for i in range(len(ts)):
        if ts[i] > br_hi[i]:
            ud.append("U")
        elif ts[i] < br_lo[i]:
            ud.append("D")
        else:
            ud.append("M")
    return ud


def udm_lengths(ud):
    ct = 0
    u = []
    d = []
    m = []
    for i in range(len(ud) - 1):
        ct += 1
        if ud[i + 1] != ud[i]:
            if ud[i] == 'U':
                u.append(ct)
            elif ud[i] == 'D':
                d.append(ct)
            else:
                m.append(ct)
            ct = 0
    return u,d,m

# ts = pd.read_csv("stkdata/TCS.csv").dropna()["Close Price"]
# br_d, br_u = get_brackets(ts, 15, 20)
# plt.plot(ts)
# plt.fill_between(range(0, len(ts)), br_d, br_u, color=(0.5, 0.5, 0.5, 0.2))
# plt.show()
# uda = ud_array(ts, br_d, br_u)
# u,d,m = udm_lengths(uda)
# print(u)
# print(d)
# print(m)
# plt.hist(u,bins=range(0,max(u)))
# plt.show()
# plt.hist(d,bins=range(0,max(d)))
# plt.show()
# plt.hist(m,bins=range(0,max(u)))
# plt.show()
