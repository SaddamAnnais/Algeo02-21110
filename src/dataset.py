import glob
import numpy as np
import cv2
import os

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

def read_image(img_list, img):
    # Read image and convert to grayscale
    n = cv2.imread(img, 0)
    # Resize the image to 256 x 256 pixels
    resized = crop_square(n, 256)
    img_list.append(resized)
    return img_list

# Get current working directory
cwd = os.getcwd()
path = glob.glob(cwd + "/dataset/*.jpg")

# Flatvector images
dataset = []
for img in path:
    list_ = []
    # Get the image name
    temp_split_name = img.split("/")
    new_temp_split_name = []
    for i in range(len(temp_split_name)):
       for j in temp_split_name[i].split("\\"):
            new_temp_split_name.append(j)
    image_name = new_temp_split_name[-1].split(".")[0]
    temp_dict = {}
    temp_dict[image_name] = read_image(list_, img)
    dataset.append(temp_dict)

# Calculate mean
mean = list(dataset[0].values())[0][0]
for i in range(1, len(dataset)):
    mean = np.add(mean, list(dataset[i].values())[0][0])
mean //= len(dataset)

# Calculate difference between training image and mean
diff = []
for i in range(len(dataset)):
    subtract_result = np.subtract(list(dataset[i].values())[0][0], mean)
    temp_dict = {}
    temp_dict[list(dataset[i].keys())[0]] = subtract_result
    diff.append(temp_dict)

# Covariant matrix
covariant = []
for i in range(len(dataset)):
    mult = np.matmul(list(dataset[i].values())[0][0], (list(dataset[i].values())[0][0]).transpose())
    if i == 0:
        covariant = mult
    else:
        np.add(covariant, mult)
covariant //= len(dataset)