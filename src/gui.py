import os
import customtkinter
import tkinter.messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import predictFace
import numpy as np
import time
import processData 
from  utils import *

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


class App(customtkinter.CTk):

    WIDTH = 1280
    HEIGHT = 720
    global click
    click = 0
    global webcamStatus
    webcamStatus = False
    global folder_imageDataSet, filename_imageRecognize
    folder_imageDataSet = None
    filename_imageRecognize = None
    global statusImage, statusDataset
    statusImage = False
    statusDataset = False

    def __init__(self):
        super().__init__()

        self.title("Tubes 2 Algeo")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  

        self.default_img = self.loadImage("src/images/default.jpg", 256)

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (12x1)
        self.frame_left.grid_rowconfigure(0, minsize=10)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure((1,4,5,8,9), weight=1)  
        self.frame_left.grid_rowconfigure(3, minsize=50)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(7, minsize=50)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(10, minsize=100)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure((11), weight=1)  
        self.frame_left.grid_rowconfigure(13, minsize=50)    # empty row with minsize as spacing


        self.switch_dataSet = customtkinter.CTkSwitch(master=self.frame_left,
                                                        text="Use Default Dataset", 
                                                        text_font=("Roboto Medium", -16),
                                                        command=lambda: self.useDefaultDataSet(self.switch_dataSet.get()))
        self.switch_dataSet.grid(row=1, column=0, columnspan = 1, padx=10, pady=50)

        self.label_dataSet = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Your DataSet",
                                              text_font=("Roboto Medium", -16)) 
        self.label_dataSet.grid(row=2, column=0 , pady=5, padx=10)

        self.button_dataSet = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose File",
                                                command=lambda: self.chooseFile(1))
        self.button_dataSet.grid(row=3, column=0, pady=5, padx=10)

        self.label_datasetStatus = customtkinter.CTkLabel(master=self.frame_left,
                                                text="Dataset not selected",
                                                text_font=("Roboto Medium", -16),
                                                fg="dark red")
        self.label_datasetStatus.grid(row=4, column=0, pady=10, padx=10)

        self.label_insertImage = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Insert Your Image",
                                              text_font=("Roboto Medium", -16)) 
        self.label_insertImage.grid(row=6, column=0,  pady=5, padx=10)

        self.button_insertImage = customtkinter.CTkButton(master=self.frame_left,
                                                text="Choose File",
                                                command=lambda: self.chooseFile(2))
        self.button_insertImage.grid(row=7, column=0, pady=5, padx=10)

        self.label_imageStatus = customtkinter.CTkLabel(master=self.frame_left,
                                                text="Image not selected",  
                                                text_font=("Roboto Medium", -16),
                                                fg="dark red")
        self.label_imageStatus.grid(row=8, column=0, pady=10, padx=10)

        self.button_compare = customtkinter.CTkButton(master=self.frame_left,
                                                text="Compare",
                                                command=self.showImage)
        self.button_compare.grid(row=10, column=0, pady=5, padx=10)

        self.label_webcam = customtkinter.CTkLabel(master=self.frame_left,
                                                    text="Webcam",
                                                    text_font=("Roboto Medium", -16))  
        self.label_webcam.grid(row=11, column=0, pady=5, padx=70,sticky='s')

        self.button_webcam = customtkinter.CTkButton(master=self.frame_left,
                                                text="Start webcam",
                                                command=self.imageWebcam)
        self.button_webcam.grid(row=12, column=0, pady=5, padx=10, sticky='s')

        # ============ frame_right ============

        # configure grid layout (4x2)
        self.frame_right.rowconfigure(1, weight=1)
        self.frame_right.rowconfigure(2, weight=0)
        self.frame_right.rowconfigure(3, weight=0)
        self.frame_right.columnconfigure((0, 1), weight=1)

        self.frame_title = customtkinter.CTkFrame(master=self.frame_right,
                                                    corner_radius=0)
        self.frame_title.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.label_title = customtkinter.CTkLabel(master=self.frame_title,
                                                text="Face Recognition",
                                                text_font=("Roboto Medium", 20)) 
        self.label_title.grid(row=0, column=1, sticky="we", pady=10, padx=10)
        
        self.frame_imageTest = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_imageTest.grid(row=1, column=0, columnspan=1, rowspan=3, pady=20, padx=20, sticky="nsew")

        self.frame_imageResult = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_imageResult.grid(row=1, column=1, columnspan=1, rowspan=3, pady=20, padx=20, sticky="nsew")

        self.frame_result = customtkinter.CTkFrame(master=self.frame_right,
                                                    highlightbackground="white",
                                                    highlightthickness=1)
        self.frame_result.grid(row=3, column=0,rowspan=2, columnspan=2, pady=50, padx=50, sticky="s")
        

        # ============ frame_image ============
        self.frame_imageTest.rowconfigure(1, weight=10)
        self.frame_imageResult.rowconfigure(1, weight=10)
        
        self.label_imageTest = customtkinter.CTkLabel(master=self.frame_imageTest,
                                                        text="Test Image",
                                                        text_font=("Roboto Medium", -16))
        self.label_imageTest.grid(row=0, column=0, pady=10, padx=20, sticky="n")

        self.image_imageTest = customtkinter.CTkLabel(master=self.frame_imageTest,
                                                       text="",
                                                       image=self.default_img )
        self.image_imageTest.grid(row=1, column=0, sticky="n")

        self.label_imageResult = customtkinter.CTkLabel(master=self.frame_imageResult,
                                                        text="Closest Result",
                                                        text_font=("Roboto Medium", -16))
        self.label_imageResult.grid(row=0, column=0, pady=10, padx=20, sticky="n")

        self.image_imageResult = customtkinter.CTkLabel(master=self.frame_imageResult,
                                                       text="", image=self.default_img)
        self.image_imageResult.grid(row=1, column=0, sticky="n")

        # configure grid layout (2x1)
        self.frame_imageTest.rowconfigure(0, weight=1)
        self.frame_imageTest.columnconfigure(0, weight=1)

        self.frame_imageResult.rowconfigure(0, weight=1)
        self.frame_imageResult.columnconfigure(0, weight=1)

        # ============ frame_result ============

        self.label_result = customtkinter.CTkLabel(master=self.frame_result,
                                                        text="Result",
                                                        text_font=("Roboto Medium", -16))
        self.label_result.grid(row=0, column=0, pady=10, padx=20)

        self.label_resultName = customtkinter.CTkLabel(master=self.frame_result,
                                                        text="None",
                                                        text_font=("Roboto Medium", -16),
                                                        fg="dark red")
        self.label_resultName.grid(row=1, column=0, pady=10, padx=20)


        self.label_time = customtkinter.CTkLabel(master=self.frame_result,
                                                        text="Execution Time",
                                                        text_font=("Roboto Medium", -16))   
        self.label_time.grid(row=0, column=1, pady=10, padx=20)

        self.label_timeValue = customtkinter.CTkLabel(master=self.frame_result,
                                                        text="0.00",
                                                        text_font=("Roboto Medium", -16),
                                                        fg="dark red")
        self.label_timeValue.grid(row=1, column=1, pady=10, padx=20)
        
    def chooseFile(self,value):
        global statusDataset, statusImage
        global folder_imageDataSet, filename_imageRecognize
        folder_imageDataSet = None
        filename_imageRecognize = None
        if value == 1:
            statusDataset = False
            folder_imageDataSet = filedialog.askdirectory(
            initialdir=cwd,
            title="Select data set directory")
            if folder_imageDataSet != "":
                statusDataset = True
                self.label_datasetStatus.configure(text="Dataset selected", fg="light green")
                self.label_imageStatus.configure(text="")
                self.image_imageResult.configure(image=self.default_img)
                self.image_imageTest.configure(image=self.default_img)
                self.label_resultName.configure(text="")
                self.label_timeValue.configure(text="")
                self.createEigenFaceGUI(folder_imageDataSet)

        elif value == 2:
            statusImage = False
            filename_imageRecognize = filedialog.askopenfilename(
            initialdir=cwd,
            filetypes=(("Image File", "*.jpg"), ("All Files", "*.*")),
            title="Select image to recognize")
            if filename_imageRecognize != "":
                statusImage = True
                self.label_imageStatus.configure(text="Image selected", fg="light green")
                global test_img
                test_img = self.loadImage(filename_imageRecognize, 256)
                self.image_imageTest.configure(image=test_img)
                self.image_imageResult.configure(image=self.default_img)

    def useDefaultDataSet(self, value):
        global statusDataset, statusImage
        global folder_imageDataSet
        if value == 1:
            statusDataset = True
            self.label_datasetStatus.configure(text="Using default dataset", fg="light green")
            self.button_dataSet.configure(state="disabled", fg_color="dark red")
            self.label_imageStatus.configure(text="")
            self.image_imageResult.configure(image=self.default_img)
            self.image_imageTest.configure(image=self.default_img)
            self.label_resultName.configure(text="")
            self.label_timeValue.configure(text="")
            self.createEigenFaceGUI(default_dataset)
        elif value == 0:
            statusDataset = False
            self.label_datasetStatus.configure(text="Dataset not selected", text_color=["gray10", "#DCE4EE"])
            self.button_dataSet.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])

    def createEigenFaceGUI(self, folderName):
        self.label_datasetStatus.configure(text="Creating eigenfaces...", fg="light blue")
        self.label_datasetStatus.update()
        start = time.time()
        processData.processDataset(folderName)
        end = time.time()
        # fungsi eigen argumen = filename_imageDataSet
        time_elapsed = end - start
        self.label_datasetStatus.configure(text=f"Eigenfaces created\n Time elapsed: {round(time_elapsed,2)}", fg="light green")

    def disabledAllButton(self):
        self.button_dataSet.configure(state="disabled", fg_color="dark red")
        self.switch_dataSet.configure(state="disabled")
        self.button_insertImage.configure(state="disabled", fg_color="dark red")
        self.button_compare.configure(state="disabled", fg_color="dark red")
    
    def enabledAllButton(self):
        # self.button_dataSet.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])
        self.switch_dataSet.configure(state="normal")
        self.button_insertImage.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])
        self.button_compare.configure(state="normal", fg_color=["#3B8ED0", "#1F6AA5"])

    # num_of_click = 0
    def imageWebcam(self):
        global click
        global webcamStatus
        global statusDataset
        click += 1
        if click % 2 == 1 and statusDataset:
            self.disabledAllButton()
            webcamStatus = True
            self.button_webcam.configure(text="Close Webcam")
            global cap
            cap = cv2.VideoCapture(0)
            self.image_imageResult.configure(image=self.default_img)
            self.frame()
        elif click % 2 == 0 and statusDataset:
            self.enabledAllButton()
            webcamStatus = False
            self.button_webcam.configure(text="Start Webcam")
            cap.release()
            self.image_imageTest.configure(image=self.default_img)
            self.image_imageResult.configure(image=self.default_img)
        else:
            click = 0
            tkinter.messagebox.showerror("Error", "Please select dataset first")



    def frame(self):
        wait = 0
        while True:
            # cv2.imshow('frame', cap.read()[1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            wait += 1
            if cap.read()[0]:
                fromCam = cv2.cvtColor(cv2.flip(cap.read()[1], 1),cv2.COLOR_BGR2RGB)
                cropped = crop_image(fromCam, 256)
                img = Image.fromarray(cropped, mode="RGB")
                imgtk = ImageTk.PhotoImage(image = img.resize((256,256)))
                self.image_imageTest.configure(image=imgtk)                    
                if wait % 40 == 0:
                    start = time.time()
                    idx = predictFace.predict(cap.read()[1], mean, E, Y, D)
                    global result_image 
                    result_image = dataset_raw[idx]
                    result_image = Image.fromarray(result_image, mode = "RGB")
                    result_image = ImageTk.PhotoImage(image = result_image)
                    self.image_imageResult.configure(image=result_image)
                    print(nama[idx])
                    self.label_resultName.configure(text=nama[idx], fg="light green")
                    end = time.time()
                    self.label_timeValue.configure(text=str(round(end-start, 2)), fg="light green")
                self.update()
        

    def loadImage(self,img_dir,img_size):
        img = Image.open(img_dir)
        img = img.resize((img_size, img_size), Image.NEAREST)
        img = ImageTk.PhotoImage(img)
        return img

    def showImage(self):
        global statusImage
        global statusDataset
        if(statusDataset != False and statusImage != False):
                # result = {dataset, idxhasil}
                start = time.time()
                idx = predictFace.predict(filename_imageRecognize, mean, E, Y, D)
                global result_image 
                result_image = dataset_raw[idx]
                result_image = Image.fromarray(result_image, mode = "RGB")
                result_image = ImageTk.PhotoImage(image = result_image)
                self.image_imageResult.configure(image=result_image)
                self.label_resultName.configure(text=nama[idx], fg="light green")
                end = time.time()
                self.label_timeValue.configure(text=str(round(end-start, 2)), fg="light green")
        else:
            tkinter.messagebox.showerror("Error", "Please select dataset and image to recognize")

    def on_closing(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
