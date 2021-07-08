import stocks

def iterate(x):
    while True:
        x(stocks.closingPrices.dropna(),stocks.get_only_name())
        if not stocks.next_stock():
            break
