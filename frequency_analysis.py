import numpy as np
import matplotlib.pyplot as plt

mymatrix = np.loadtxt('f237',delimiter=',',skiprows=2)
# mymatrix[:,0] = (mymatrix[:,0] - 5361.)
mymatrix[:,0] = (mymatrix[:,0])
s = mymatrix[:,0]

print(s)

t = np.linspace(0,.005,np.shape(s)[0])

# plt.ylabel("Amplitude")
# plt.xlabel("Time [s]")
# plt.plot(t, s)
# plt.show()



t = np.linspace(0, 0.05, 500)
s = np.sin(40 * 2 * np.pi * t)

plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.plot(t, s)
plt.show()


fft = np.fft.fft(s)
T = t[1] - t[0]  # sampling interval
N = s.size

# 1/T = frequency
f = np.linspace(0, 1 / T, N)

print(f)
print("hello")
print(fft*1/N)

plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor
plt.show()