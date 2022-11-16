import numpy as np
import copy

def qr(A):
    len = A.shape[0]
    Q = np.identity(len)
    for i in range(len):
        H = np.identity(len)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A

def make_householder(a):
    e = np.zeros_like(a)
    e[0] = 1
    v = a + np.copysign(np.linalg.norm(a), a[0])*e
    H = np.identity(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def getEigen(matrix, tolerance):
    # mengembalikan eigen value dan eigenvector menggunakan QR algorithm
    E = copy.deepcopy(matrix)
    q,r = qr(E); v = q; count = 1
    
    while True:
        prevDiag = np.diagonal(E)
        E = np.matmul(r,q)
        q,r = qr(E)
        
        count+=1
        v = np.matmul(v,q)

        currentDiag = np.diagonal(E)

        if count % 200 == 0:
            print(count)
        if np.allclose(prevDiag, currentDiag, atol=tolerance) or count == 4000 :
            print(count)
            # print(E)
            break
    return currentDiag, v