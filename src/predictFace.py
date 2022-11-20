import cv2
import numpy as np
from utils import *

nama = np.load('src/nama.npy')
mean = np.load('src/mean.npy')
E = np.load('src/E.npy')
Y = np.load('src/Y.npy')
D = np.load('src/D.npy')
dataset = np.load('src/dataset.npy')

# Predict new face
def predict(dirGambar, mean, E, Y, D): 
    D = D[0]
    newFace = read_image(dirGambar)
    # cv2.imshow("test", newFace.reshape((256,256)))
    # cv2.waitKey(0)

    newFace = (newFace - mean).reshape(256*256,1)
    newWeight = E.T @ newFace

    # Mencari yg paling mirip menggunakan eucledian distance
    weightSelisih = newWeight - Y[:,0].reshape(D,1)
    selisih = eucledian(weightSelisih)
    idxHasil = 0

    for i in range(1, Y.shape[1]):
        weightSelisih = newWeight - Y[:,i].reshape(D,1)
        tempSelisih = eucledian(weightSelisih)
        if selisih > tempSelisih :
            idxHasil = i
            selisih = tempSelisih
            print(selisih)
    print()        
    if selisih>=800000:
        return -1    
    else :
        return idxHasil

# if __name__ == "__main__":
#     predict("test\Emma Watson88_2041.jpg", mean, E, Y, D, dataset, nama)
