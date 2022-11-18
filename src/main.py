import customtkinter
import gui
import os
import numpy as np

customtkinter.set_appearance_mode("Dark") 
customtkinter.set_default_color_theme("blue")  

cwd = os.getcwd()
nama = np.load('src/nama.npy')
mean = np.load('src/mean.npy')
E = np.load('src/E.npy')
Y = np.load('src/Y.npy')
D = np.load('src/D.npy')
dataset_raw = np.load('src/dataset_raw.npy')
default_dataset = "dataset"


if __name__ == "__main__":
    app = gui.App()
    app.mainloop()
