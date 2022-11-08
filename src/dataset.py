import glob
import numpy as np
import cv2
import os

def read_image(img_list, img):
    # Read image and convert to grayscale
    n = cv2.imread(img, 0)
    # Resize the image to 200 x 200 pixels
    resized = cv2.resize(n, (200, 200), interpolation=cv2.INTER_AREA)
    img_list.append(resized)
    return img_list

# Get current working directory
cwd = os.getcwd()
path = glob.glob(cwd + "/dataset/*.jpg")

# Flatvector images
list_ = []
dataset = [read_image(list_, img) for img in path]

# Calculate mean
mean = dataset[0]
for i in range(1, len(dataset)):
    mean = np.add(mean, dataset[i])
mean //= len(dataset)

# Calculate difference between training image and mean
diff = []
for i in range(len(dataset)):
    diff.append(np.subtract(dataset[0], mean))