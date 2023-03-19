import pickle
import numpy as np

name = './EXP2'

X_tra, y_tra, X_tst, y_tst = np.array(pickle.load(open(name + '.pickle','rb')),dtype=object)

x_training = np.zeros((1,1))
y_training = np.zeros((1,1))
x_testing = np.zeros((1,1))
y_testing = np.zeros((1,1))

for i in range(14):
    
    remand = 120
    remand_tst = 64
    if i == 0:
        index = (y_tra == i)
        X_temp = X_tra[index]
        shuffle = np.random.permutation(X_temp.shape[0])
        x_training = X_temp[shuffle][:remand]
        y_training = np.array([i for n in range(remand)])

        # index = (y_tst == i)
        # X_temp = X_tst[index]
        # shuffle = np.random.permutation(X_temp.shape[0])
        x_testing = X_temp[shuffle][remand:remand+remand_tst]
        y_testing = np.array([i for n in range(remand_tst)])
        
    else:
        index = (y_tra == i)
        X_temp = X_tra[index]
        shuffle = np.random.permutation(X_temp.shape[0])
        x_training = np.concatenate((x_training,X_temp[shuffle][:remand]),axis=0)
        y_training = np.concatenate((y_training,np.array([i for n in range(remand)])),axis=0)

        # index = (y_tst == i)
        # X_temp = X_tst[index]
        # shuffle = np.random.permutation(X_temp.shape[0])
        x_testing = np.concatenate((x_testing,X_temp[shuffle][remand:remand+remand_tst]),axis=0)
        y_testing = np.concatenate((y_testing,np.array([i for n in range(remand_tst)])),axis=0)
        


data = {'data':x_training,'label':y_training}
np.save(name + '_training.npy',data)
print(x_training.shape)
print(y_training.shape)

data = {'data':x_testing,'label':y_testing}
np.save(name + '_testing.npy',data)
print(x_testing.shape)
print(y_testing.shape)

# data = {'data':np.concatenate((X,X_tst)),'label':np.concatenate((Y,y_tst))}
# np.save(name + '_full.npy',data)
# print(data['data'].shape)