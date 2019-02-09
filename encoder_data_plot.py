import numpy as np
import matplotlib.pyplot as plt

time_data_collect = 49928.
number_data_points = 1700

Y0=np.loadtxt('one_thousand_position_readings')
# Y0=position_data[:]

Y1=np.loadtxt('one_thousand_velocity_readings')
# Y1=velocity_data[:]

X = np.linspace(0.,time_data_collect,number_data_points)
# X = t[:]

plt.figure(0)
plt.plot(X,Y0,':ro')
# plt.ylim((0,55000))
# plt.show() #or
plt.savefig('positions.png')

plt.figure(1)
plt.plot(X,Y1,':go')
plt.savefig('velocities.png')
plt.show()