import numpy as np
import copy
import cv2
import mediapipe as mp



def eucledian(array):
    sum = np.sum(np.square(array))
    return np.sqrt(sum)

def normalize(array):
    normalizedArray = (array-np.min(array))/(np.max(array)-np.min(array))
    return normalizedArray

def crop_image(img, size):
    # input gambar berwarna, output gambar berwarna
    # mengcropped image sehingga hanya tampil bagian mukanya saja
    img2 = copy.deepcopy(img)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    if (isinstance(faces, np.ndarray)):     # jika wajah terdeteksi, maka dicrop
        for (x, y, w, h) in faces:
            img2 = img[y:y+h,x:x+w]
    resized = cv2.resize(img2, (size, size), interpolation=cv2.INTER_AREA)
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
    bg_image = np.zeros(shape=(img.shape[0],img.shape[1]))+255
    output_image = np.where(conditionMask,img,bg_image).astype(np.uint8)
    if (output_image == bg_image).all():
        return img
    else :
        return output_image

def read_image(imgIn):
    if type(imgIn) == str:
        imgIn = cv2.imread(imgIn)                 # Read image
    
    noBgImage = remove_bg(imgIn)                # Remove background from image
    processedImage = crop_image(noBgImage, 256)  # Crop to face and resize it to (256, 256)
    # processedImage = crop_square(noBgImage, 256)   # Crop image and resize it to (256, 256)
    return processedImage.reshape(1,256*256)                # Matrix with shape (1, 256x256)