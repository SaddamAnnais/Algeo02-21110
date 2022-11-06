import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# define windows
root = Tk()
root.title = "Tubes 2 Algeo"
root.geometry("1280x720")

cwd = os.getcwd()

# define function
def chooseFile(value):
    global filename_imageDataSet, filename_imageRecognize
    if value == 1:
        filename_imageDataSet = filedialog.askopenfilenames(
        initialdir=cwd,
        filetypes=(("Image File", "*.jpg"), ("All Files", "*.*")),
        title="Select data set directory")
    elif value == 2:
        filename_imageRecognize = filedialog.askopenfilename(
        initialdir=cwd,
        filetypes=(("Image File", "*.jpg"), ("All Files", "*.*")),
        title="Select image to recognize")
    showImage(value)


def showImage(value):
    if value == 1:
        label_testImage.destroy()
        # get image
        imageDataSet = Image.open(filename_imageDataSet[0])
        #resize image
        imageDataSet = imageDataSet.resize((200, 200), Image.NEAREST)
        # convert image to Tkinter format
        imageDataSet = ImageTk.PhotoImage(imageDataSet)
        # show image
        label_imageDataSet = Label(frame_testImage, image=imageDataSet)
        label_imageDataSet.image = imageDataSet
        label_imageDataSet.grid(row=1, column=0)
    elif value == 2:
        label_resultImage.destroy()
        # get image
        imageResult = Image.open(filename_imageRecognize)
        # resize image
        imageResult = imageResult.resize((200, 200) , Image.NEAREST)
        # convert image to Tkinter format
        imageResult = ImageTk.PhotoImage(imageResult)
        # show image
        label_imageRecognize = Label(frame_resultImage, image=imageResult)
        label_imageRecognize.image = imageResult
        label_imageRecognize.grid(row=1, column=0)


# define label
Title = Label(root, text="Face Recognition", font=("Arial", 30))

# define frame and label
frame_dataSet = LabelFrame(
    root, text="Insert Your DataSet", 
    font=("Arial", 12), 
    padx=100, pady=10)

frame_recognize = LabelFrame(
    root, text="Insert Your Image", 
    font=("Arial", 12), 
    padx=100, pady=10)

frame_result = LabelFrame(
    root, text="Result", 
    font=("Arial", 12), 
    padx=100, pady=10)


label_result = Label(
    frame_result, text="Result", 
    font=("Arial", 12))

frame_time = LabelFrame(
    root, text="Execution Time", 
    font=("Arial", 12), 
    padx=100, pady=10)

label_time = Label(
    frame_time, text="Time", 
    font=("Arial", 12))


frame_testImage = LabelFrame(
    root, text="Test Image", 
    font=("Arial", 12), 
    padx=100, pady=100)

label_testImage = Label(
    frame_testImage, text="Test Image", 
    font=("Arial", 12))

frame_resultImage = LabelFrame(
    root, text="Closest Result", 
    font=("Arial", 12), 
    padx=100, pady=100)

label_resultImage = Label(
    frame_resultImage, text="Closest Result",
     font=("Arial", 12))

# define buttons
button_dataSet = Button(
    frame_dataSet, text="Choose File",
     font=("Arial", 12),
     command=lambda: chooseFile(1))


button_recognize = Button(
    frame_recognize, text="Choose File", 
    font=("Arial", 12), 
    command=lambda: chooseFile(2))

# define grid layout
Title.grid(row=0, column=4, columnspan=9)

# dataSet
frame_dataSet.grid(row=1, column=0, padx=10, pady=10)
button_dataSet.grid(row=2, column=0)

# recognize
frame_recognize.grid(row=3, column=0, padx=10, pady=10)
button_recognize.grid(row=4, column=0)

# result
frame_result.grid(row=5, column=0)
label_result.grid(row=6, column=0)

# time
frame_time.grid(row=5, column=4)
label_time.grid(row=6, column=4)

# test image
frame_testImage.grid(row=1, column=4, rowspan=4, columnspan=2)
label_testImage.grid(row=1, column=4)

# result image
frame_resultImage.grid(row=1, column=7, rowspan=4, columnspan=2)
label_resultImage.grid(row=1, column=7)

root.mainloop()