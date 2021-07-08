from PyPDF2 import PdfFileReader, PdfFileWriter
import matplotlib.pyplot as plt


class PDF:
    def __init__(self, name):
        self.rr = None
        self.wr = PdfFileWriter()
        self.name = name

    def add(self, add=True, show=False):
        if not add:
            plt.show()
            return
        plt.savefig("pdf/temp.pdf")
        if show:
            plt.show()
        self.rr = PdfFileReader("pdf/temp.pdf")
        self.wr.addPage(self.rr.getPage(0))

    def save(self):
        fl = open(self.name, "bw")
        self.wr.write(fl)
