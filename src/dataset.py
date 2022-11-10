import glob
import numpy as np
import cv2
import os
import mediapipe as mp
from eigen import getEigen 

def crop_image(img, size):
    # input gambar berwarna, output gambar berwarna
    # mengcropped image sehingga hanya tampil bagian mukanya saja
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    if (isinstance(faces, np.ndarray)):     # jika wajah terdeteksi, maka dicrop
        for (x, y, w, h) in faces:
            img = img[y:y+h,x:x+w]
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return resized

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

def remove_bg(img):
    # Menghilangkan background dari gambar sehingga training jadi lebih akurat
    # Input gambar berwarna ukuran (256,256,3)
    # Output gambar grayscale ukuran (256,256)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(img)

    mask = results.segmentation_mask
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    conditionMask = mask > 0.5
    bg_image = np.zeros(shape=(img.shape[0],img.shape[1]))
    output_image = np.where(conditionMask,img,bg_image).astype(np.uint8)
    if (output_image == bg_image).all():
        return img
    else :
        return output_image

def read_image(img_dir):
    rawImage = cv2.imread(img_dir)                 # Read image
    noBgImage = remove_bg(rawImage)                # Remove background from image
    # processedImage = crop_image(noBgImage, 256)      # Crop raw image and resize it to (256, 256,3)
    processedImage = crop_square(noBgImage, 256)
    return processedImage

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
    temp_dict[image_name] = read_image(img)
    dataset.append(temp_dict)

# Calculate mean
mean = np.zeros(shape=(256,256))
for i in range(0, len(dataset)):
    mean = np.add(mean, list(dataset[i].values())[0])
mean = np.divide(mean, len(dataset)).astype(np.uint8)

# Calculate difference between training image and mean
diff = []
for i in range(len(dataset)):
    subtract_result = np.subtract(list(dataset[i].values())[0], mean)
    temp_dict = {}
    temp_dict[list(dataset[i].keys())[0]] = subtract_result
    diff.append(temp_dict)

# Covariant matrix
A = list(diff[0].values())[0]
for i in range(1, len(dataset)):
    A = np.append(A, list(diff[i].values())[0], axis=1)
covariant = np.matmul(A, A.T)
# print(np.shape(covariant))

# Get eigenvector
# eigval, eigvector = getEigen(covariant)
eigval, eigvector = np.linalg.eig(covariant)
eigvector = np.real(eigvector)

# Calculate eigenface. Eigenface = v x (dataset[i] - mean[i])
eigFaceMat = [np.matmul(eigvector, list(diff[0].values())[0])]
for i in range(1, len(dataset)):
    eigFace = np.matmul(eigvector, list(diff[i].values())[0])
    eigFaceMat = np.append(eigFaceMat, [eigFace], axis=0)

# Predict new face
newFace = read_image("gal gadot85_1842.jpg")
cv2.imshow("test", newFace)
cv2.waitKey(0)

newFace = newFace - mean
newFace = np.matmul(eigvector, newFace)

matSelisih = newFace - eigFaceMat[0]
selisih = np.sqrt(np.sum(np.square(matSelisih)))
idxHasil = 0

# print(np.shape(matSelisih))
# print(selisih)
for i in range(1, len(eigFaceMat)):
    matSelisih = newFace - eigFaceMat[i]
    if selisih > np.sqrt(np.sum(np.square(matSelisih))) :
        idxHasil = i
        selisih = np.sqrt(np.sum(np.square(matSelisih)))
        print(selisih)

matHasil = list(dataset[idxHasil].values())[0]
print(list(dataset[idxHasil].keys())[0])
print(idxHasil)
cv2.imshow("hasil", matHasil)
cv2.waitKey(0)    