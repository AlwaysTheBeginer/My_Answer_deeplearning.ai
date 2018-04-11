import h5py as hdf5
from PIL import Image
import numpy as np

def  sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

trainDataset=hdf5.File("./datasets/train_catvnoncat.h5", "r")

keyList=trainDataset.keys()
keyNameList=[]
valueList=[]
print(keyList)
for key in keyList:
    keyNameList.append(trainDataset[key].name)
    valueList.append(trainDataset[key].value)
print(keyNameList)

train_set_x_shape=valueList[1].shape
print(train_set_x_shape)
m=train_set_x_shape[0]

n_x=64*64*3
dw=np.zeros((n_x,1))
W=np.zeros((n_x,1))
X=np.zeros((n_x,0))
for i in range(0,m):
    r,g,b=Image.fromarray(valueList[1][i],mode='RGB').split()
    r=np.asmatrix(r).reshape(int(n_x/3),1)
    g=np.asmatrix(g).reshape(int(n_x/3), 1)
    b=np.asmatrix(b).reshape(int(n_x/3), 1)
    x=np.row_stack((r,g,b))
    X =np.column_stack((X, x))
B=0
X=X.reshape(n_x,m)
W=W.reshape(n_x,1)

alpha=0.05

Y=np.matrix(valueList[2])
for i in range(0,500):
    if(i%20==0):
        print("Iterate:"+str(i)+"\n")
    Z=W.T*X+B
    A=sigmoid(Z)
    dZ=A-Y
    dW=1/m*X*dZ.T
    dB=1/m*np.sum(dZ,axis=1)
    W-=alpha*dW
    B-=alpha*dB

parameterFile=open("parameterW.dat","wb+")
parameterFile.write(W)
parameterFile.close()
parameterFile=open("parameterB.dat","wb+")
parameterFile.write(B)
parameterFile.close()