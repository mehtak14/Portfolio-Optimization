class Cover:
    def __init__(self, closingPrices, coverWidth):
        self.closingPrices = list(closingPrices)
        self.coverWidth = coverWidth
        self.start = 0
        self.end = coverWidth - 1
        self.min = min(closingPrices[0:coverWidth])
        self.max = max(closingPrices[0:coverWidth])
        self.delta = closingPrices[self.end] - closingPrices[self.end - 1]
        self.curr = closingPrices[self.end]

    def move_forward(self):
        if self.end < len(self.closingPrices) - 1:
            if self.closingPrices[self.start] == self.min:
                self.min = min(self.closingPrices[self.start + 1:self.end + 1])
            if self.closingPrices[self.start] == self.max:
                self.max = max(self.closingPrices[self.start + 1:self.end + 1])
            self.start = self.start + 1
            if self.closingPrices[self.end + 1] < self.min:
                self.min = self.closingPrices[self.end + 1]
            if self.closingPrices[self.end + 1] > self.max:
                self.max = self.closingPrices[self.end + 1]
            self.end = self.end + 1
            self.delta = self.closingPrices[self.end] - self.closingPrices[self.end - 1]
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
