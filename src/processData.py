import glob
import numpy as np
import cv2
import os
from eigen import getEigen 
import copy
from utils import *

def processDataset(folderName):
    # Get current working directory
    # cwd = os.getcwd()
    path = glob.glob(folderName + "/*.jpg")

    # Flatvector images
    dataset_raw = []
    for img in path:
        raw_img = cv2.imread(img)
        raw_img = cv2.resize(raw_img, (256, 256))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        dataset_raw.append(raw_img)

    # Flatvector images
    dataset = []
    nama = []
    for img in path:
        list_ = []
        # Get the image name
        temp_split_name = img.split("/")
        new_temp_split_name = []
        for i in range(len(temp_split_name)):
            for j in temp_split_name[i].split("\\"):
                new_temp_split_name.append(j)
        image_name = new_temp_split_name[-1].split(".")[0]
        nama.append(image_name)
        dataset.append(read_image(img))

    nama = np.array(nama)
    dataset = np.array(dataset)

    # Calculate mean
    mean = np.zeros(shape=(1,256*256))
    for i in range(0, len(dataset)):
        mean = np.add(mean, dataset[i])
    mean = np.divide(mean, len(dataset)).astype(np.uint8)

    cv2.imshow("mean", mean.reshape((256,256)))
    cv2.waitKey(0)

    # Calculate difference between training image and mean
    M = dataset[0].reshape(256*256,1)
    for i in range(1, len(dataset)):
        M = np.append(M, dataset[i].reshape(256*256,1), axis = 1)
    A = (M - mean.T).astype(np.uint8)

    cv2.imshow("Normalised", A[:,0].reshape((256,256)))
    cv2.waitKey(0)

    # Calculate eigen values and eigen vectors of covariant matrix
    ## Since covariant = A @ AT that have a shape of (256*256, 256*256), not computationally efficient 
    ## We first compute AT @ A that have a shape of (M, M) M : Number of sample in dataset
    ## After that we need to compute A @ eigvec to get the largest M eigenvectors of covariant
    cov = np.cov(A.T)
    eigval, eigvec = np.linalg.eig(cov)
    # eigval, eigvec = np.linalg.eig(A.T @ A)
    eigvec = A @ eigvec

    ## 90 % of the total variance of eigenvectors contain in the first 10% of the largest corresponding eigenvalue
    ## Because of that, we need to sort the eigenvalues and eigenvectors
    ## After that, we need to take 25% (rule of thumb) of all the eigenvector 
    eigpairs = [(eigval[idx], eigvec[:,idx]) for idx in range(len(eigval))]

    D = (25 * len(eigpairs))//100
    # D = len(eigpairs)

    eigpairs.sort(reverse=True)
    eigval_sort = [eigpairs[idx][0] for idx in range(D)]
    eigvec_sort = [eigpairs[idx][1] for idx in range(D)]

    eigval_sort = np.array(eigval_sort)
    eigvec_sort = np.array(eigvec_sort).T
    # print(eigvec_sort)

    # Normalize
    E = copy.copy(eigvec_sort)
    for i in range(E.shape[1]):
        E[:,i] = normalize(E[:,i])
    # E = E*255
    # print(E[:,1])
    # E = E.astype(np.uint8)

    # Calculating weight of train image
    Y = E.T @ A

    cv2.imshow("Eigface1", E[:,0].reshape(256,256))
    cv2.imshow("Eigface2", E[:,2].reshape(256,256))
    cv2.imshow("Eigface3", E[:,3].reshape(256,256))
    cv2.imshow("Eigface-2", E[:,D-2].reshape(256,256))
    cv2.imshow("Eigface-1", E[:,D-1].reshape(256,256))
    cv2.waitKey(0)

    # Calculating weight of train image
    Y = E.T @ A

    D = np.array([D])
    print("Training is completed")

    np.save('src/Y.npy', Y)
    np.save('src/E.npy', E)
    np.save('src/nama.npy', nama)
    np.save('src/dataset.npy', dataset)
    np.save('src/mean.npy', mean)
    np.save('src/D.npy', D)
    np.save('src/dataset_raw.npy', dataset_raw)

if __name__ == "__main__":
    processDataset("dataset")