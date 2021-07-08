import tkinter as tk
import stock_gui
import markov_gui
import pdf
import os


### Load Window ##########################

def fill_list():
    for x in files:
        lb.insert(tk.END, x.split(".")[0])


def list_click(x):
    # stock_gui.StockGUI(lb.get(lb.curselection())).launch()
    markov_gui.MarGui(lb.get(lb.curselection())).launch()


win = tk.Tk()
win.title("PortfolioOptimization")
screen_height = win.winfo_screenheight()
screen_width = win.winfo_screenwidth()
win.minsize(int(screen_width / 5), int(screen_height / 3))
lb = tk.Listbox(win)
files = os.listdir("stkdata/")
lb.configure(justify=tk.CENTER)
fill_list()
lb.bind("<Double-1>", list_click)
lb.pack()

# --------------------------------

pdf_frame = tk.Frame(win)
pdf_frame.pack()

pdf_name_history = "temp.pdf"
pdf_file = pdf.PDF("pdf/" + pdf_name_history)
pdf_name = tk.Entry(pdf_frame)
pdf_name.insert(0, pdf_name_history)
pdf_name.pack(side=tk.LEFT)


def pdf_add():
    global pdf_file, pdf_name_history, pdf_name
    if pdf_name_history == pdf_name.get():
        pdf_file.add()
        pdf_file.save()
    else:
        pdf_name_history = pdf_name.get()
        pdf_file = pdf.PDF("pdf/" + pdf_name.get())
        pdf_file.add()
        pdf_file.save()


add_btn = tk.Button(pdf_frame, text="Add current plot", command=pdf_add)
add_btn.pack(side=tk.LEFT)

##########################################

win.mainloop()
