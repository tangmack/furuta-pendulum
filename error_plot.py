import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import sepfir2d

# TO TRY: SUBTRACT THE DERIVATIVE OF THE PROPORTIONAL ERROR, ADD
# mymatrix = np.loadtxt('f255',delimiter=',',skiprows=2)
# mymatrix = np.loadtxt('a412',delimiter=',',skiprows=2)
# mymatrix = np.loadtxt('pgain00285dc015left',delimiter=',',skiprows=2)
mymatrix = np.loadtxt('vid',delimiter=',',skiprows=2)
print(mymatrix)

mymatrix[:,0] = (mymatrix[:,0] - 5361.)/200
# mymatrix[:,0] = sepfir2d(mymatrix[:,0], 2, 0)

a = np.fft.fft(mymatrix[:,0])


# mymatrix[:,2] = mymatrix[:,2]/0.0003834951969714103074295218974/0.15/1000
# mymatrix[:,2] = mymatrix[:,2]
# a = mymatrix[:,2]
# b = np.reshape(a,(np.shape(a)[0],1))
#
# # mymatrix[:,:-1] = np.transpose(mymatrix[:,2])
# print(np.transpose(mymatrix[:,2]))
# print(np.shape(a))
# print(np.shape(np.transpose(a)))
# print(np.shape(b))

# newmatrix = np.append(mymatrix,b,axis=1)
# print(newmatrix)

# print(newmatrix)
plt.axhline(y=0, color='r', linestyle='-')
# plt.legend("ah")
plt.plot(mymatrix)
plt.show()
plt.savefig('vid.png')