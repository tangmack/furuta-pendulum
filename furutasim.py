"""
===========================
Furuta Pendulum Simulation
===========================

This animation illustrates the Furuta Pendulum problem.
"""

# Code modified from Double Pendulum animation example code found here:
# https://matplotlib.org/examples/animation/double_pendulum_animated.html

# Dynamics and State Space Representation found from paper:
# "Nonlinear stabilization control of Furuta pendulum only
# using angle position measurements"
# Authors: Lin Zhao, Shuli Gong, Ancai Zhang, Lanmei Cong

import time
t0 = time.time()

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
# import cv2

# import math


G = 9.8  # acceleration due to gravity, in m/s^2

# Motor
M1 = .0695  # mass of pendulum 1 in kg
L1 = .035  # length of pendulum 1 in m
l1 = 0.025  # radius of center of mass
J1 = 0.00015625
B1 = 0.0

# Pendulum
M2 = .025  # mass of pendulum 2 in kg
L2 = 0.06  # length of pendulum 2 in m
l2 = 0.03  # radius
J2 = 0.00003
B2 = 0.0
# FRIC1 = .3  # friction coefficient (* by velocity)
# FRIC2 = .3  # friction coefficient (* by velocity)

# A1 = J1 + M1 * R1 * R1
# A2 = M1 * R1 * L2
# A3 = J2 + M2 * R2 * R2 + M1 * L2 * L2
# A4 = M1 * G * R1

A0 = J1 + M1 * l1 * l1 + M2 * L1 * L1
A2 = J2 + M2 * l2 * l2

peak_torque_output = 0.0  # track maximum torque output from motor
time_to_45degree_error = np.inf  # track time to get nearly upright


def derivs(state, t):

    dydx = np.zeros_like(state)

    global peak_torque_output
    global time_to_45degree_error

    # if (t<20.0):
    # if (state[2] > 2.96706) or (state[2] < -2.96706):
    # if ((np.pi - abs(state[2])) * np.sign(state[2]) * .05) > 0:
        # tau1 = .05
    # elif ((np.pi - abs(state[2])) * np.sign(state[2]) * .05) < 0:
        # tau1 = -.05
    tau1 = (np.pi - abs(state[2])) * np.sign(state[2]) * .18 - 0.004 * state[3]
    # tau1 = (np.pi - abs(state[2])) * np.sign(state[2]) * .18
    # tau1 = 0.0
    # tau1 = 0.0
    # else:
    #     tau1 = -state[3] * 0.001 # motor torque

    # if t < time_to_45degree_error:
    if (abs(tau1) > abs(peak_torque_output)):
            peak_torque_output = tau1

    if (state[2] > 2.35619) or (state[2] < -2.35619):
        time_to_45degree_error = t

    # tau1 = 0.0
    # else:
        # tau1 = 0.0
    # tau2 = 0.0  # disturbance torque (system not actuated here)!
    tau2 = -.00001 * state[3] # apply friction

    mytwos = 2 * state[2]


    dydx[0] = state[1]
    dydx[1] = (2 * A2 * tau1 - 2 * A2 * B1 * state[1] - 2 * L1 * l2 * M2 * tau2 * cos(state[2]) \
            + 2 * B2 * L1 * l2 * M2 * state[3] * cos(state[2]) \
            + 2 * A2 * L1 * l2 * M2 * state[3] * state[3] * sin(state[2]) \
            + 2 * G * L1 * l2 * l2 * M2 * M2 * cos(state[2]) * sin(state[2]) \
            - 2 * A2 * A2 * state[1] * state[3] * sin( mytwos ) \
            - A2 * L1 * l2 * M2 * state[1] * state[1] * cos(state[2]) * sin(mytwos)) \
            / ( 2 * ( A0 * A2 - L1 * L1 * l2 * l2 * M2 * M2 * cos(state[2]) * cos(state[2]) + A2 * A2 * sin(state[2]) * sin(state[2]) ) )

    dydx[2] = state[3]
    dydx[3] = (2 * A0 * tau2 \
            - 2 * A0 * B2 * state[2] \
            - 2 * L1 * l2 * M2 * tau1 * cos(state[2]) \
            + 2 * B1 * L1 * l2 * M2 * state[1] * cos(state[2]) \
            - 2 * A0 * G * l2 * M2 * sin(state[2]) \
            - 2 * L1 * L1 * l2 * l2 * M2 * M2 * state[3] * state[3] * cos(state[2]) * sin(state[2]) \
            + 2 * A2 * tau2 * sin(state[2]) * sin(state[2]) \
            - 2 * A2 * B2 * state[3] * sin(state[2]) * sin(state[2])  \
            - 2 * A2 * B2 * state[3] * sin(state[2]) * sin(state[2]) \
            - 2 * A2 * G * l2 * M2 * sin(state[2]) * sin(state[2]) * sin(state[2])
            + A0 * A2 * state[1] * state[1] * sin(mytwos) \
            + 2 * A2 * L1 * l2 * M2 * state[1] * state[3] * cos(state[2]) * sin(mytwos) \
            + A2 * A2 * state[1] * state[1] * sin(state[2]) * sin(state[2]) * sin(mytwos) ) \
            / ( 2 * (A0 * A2 - L1 * L1 * l2 * l2 * M2 * M2 * cos(state[2]) * cos(state[2]) + A2 * A2 * sin(state[2]) * sin(state[2]) ) )

    return dydx

# create a time array from 0..100 sampled at 0.01 second steps
# dt = 0.03125
dt = 0.01428571428
t = np.arange(0.0, 15.0, dt)
myfps = int(1/dt)
myinterval = int(dt*1000)

# Initial conditions
q1 = 180.0  # angle of motor
q1d = 0.0  # initial angular speed of motor

q2 = 177.0  # angle of pendulum
q2d = 0.0  # initial angular speed of pendulum


# X1, X2, X3, X4 are the state space representation
# X1 = q1
# X2 = A1 * q1d - A2 * cos(math.radians(q1) * math.radians(q2d))
# X3 = q2
# X4 = q2d

# initial state
state = np.radians([q1, q1d, q2, q2d])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = -L1*sin(y[:, 0])
y1 = L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2])
y2 = -L2*cos(y[:, 2])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1))
fig.gca().set_aspect('equal',adjustable='box')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [0, x1[i]]
    thisy = [0, y1[i]]

    thisx2 = [0, x2[i]]
    thisy2 = [0, y2[i]]

    line.set_data(thisx, thisy)
    line2.set_data(thisx2,thisy2)
    time_text.set_text(time_template % (i*dt))
    return line, line2, time_text


ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=0, blit=False, init_func=init)

# Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani.save('f3.mp4', fps=myfps, dpi=60, writer='ffmpeg')
# plt.show()

print("peak torque used was ", peak_torque_output, " Nm.")
print("time to reach less than 45 degree error: ", time_to_45degree_error)

t1 = time.time()
print(t1-t0)
