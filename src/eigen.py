import numpy as np
import copy

testArr = np.random.randint(0,256, size=(256,256))#/np.random.randint(1,500,size=(2,2))
# testArr = np.array(([3,2,3,2],
#                     [5,1,5,4],
#                     [9,3,2,1],
#                     [4,5,6,7]))

# testArr = np.array(([158,235,243],
#                     [238,64,40],
#                     [164,182,143]))

# testArr = testArr @ np.transpose(testArr)
def getEigen(A):
    # mengembalikan eigen value dan eigenvector menggunakan QR algorithm
    E = copy.deepcopy(A)
    q,r = np.linalg.qr(E); v = q; count = 1
    
    while True:
        prevDiag = np.diagonal(E)
        E = np.matmul(r,q)
        q,r = np.linalg.qr(E)
        
        count+=1
        v = np.matmul(v,q)

        currentDiag = np.diagonal(E)

        if count % 200 == 0:
            print(count)
        if np.allclose(prevDiag, currentDiag, atol=0.01) or count == 1500 :
            print(count)
            # print(E)
            break
    

    return currentDiag, v

# print(testArr)
print(" \n")
# print(testArr)
q, vec = getEigen(testArr)
# print(np.sum(q))
# print(q)
print(np.sum(vec))
# print(vec)

val, vector = np.linalg.eig(testArr)
# print(np.sum(val))
# print(val)
print(np.sum(vector))
# print(vector)