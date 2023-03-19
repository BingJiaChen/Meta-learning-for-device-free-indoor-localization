import numpy as np
import matplotlib.pyplot as plt


c1 = np.load('./dataset/50_val.npy')[:60]
c2 = np.load('./dataset/100_val.npy')[:30]
c3 = np.load('./dataset/150_val.npy')[:20]
c4 = np.load('./dataset/200_val.npy')[:15]
cen = np.load('./dataset/central.npy')[:30]

x1 = (np.arange(len(c1))+1) * 50
x2 = (np.arange(len(c2))+1) * 100
x3 = (np.arange(len(c3))+1) * 150
x4 = (np.arange(len(c4))+1) * 200
x_cen = (np.arange(len(cen))+1) * 100

print(np.max(c2))

plt.figure()
plt.plot(x1,c1,'b-',x2,c2,'r-',x3,c3,'g-',x4,c4,'y-',x_cen,cen)
plt.legend(['Federated meta-learning (n_local=50)','Federated meta-learning (n_local=100)','Federated meta-learning (n_local=150)','Federated meta-learning (n_local=200)','Centralized'])
plt.xlabel('Time')
plt.ylabel('Classification accuracy (%)')
plt.show()






