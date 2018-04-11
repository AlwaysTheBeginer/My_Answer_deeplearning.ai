import h5py as hdf5
from PIL import Image
import numpy as np

def  sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

testDataset=hdf5.File("./datasets/test_catvnoncat.h5", "r")

keyList=testDataset.keys()
keyNameList=[]
valueList=[]
print(keyList)
for key in keyList:
    keyNameList.append(testDataset[key].name)
    valueList.append(testDataset[key].value)
print(keyNameList)

test_set_x_shape=valueList[1].shape

n_x=64*64*3
m=test_set_x_shape[0]

X=np.zeros((n_x,0))
for i in range(0,m):
    r,g,b=Image.fromarray(valueList[1][i],mode='RGB').split()
    r=np.asmatrix(r).reshape(int(n_x/3),1)
    g=np.asmatrix(g).reshape(int(n_x/3), 1)
    b=np.asmatrix(b).reshape(int(n_x/3), 1)
    x=np.row_stack((r,g,b))
    X =np.column_stack((X, x))
X=X.reshape(n_x,m)

W=np.fromfile("parameterW.dat").reshape(n_x,1)
B=np.fromfile("parameterB.dat")
Z=W.T*X+B
Y_hat=sigmoid(Z)
Y_hat=np.round(Y_hat)
print(Y_hat)
Y=np.matrix(valueList[2])
print(Y)
error=np.count_nonzero(np.abs(Y-Y_hat))
print("error:"+str(error)+" in total:"+str(m))
accuracy=1-error/m
print("Accuracy:"+str(accuracy))
