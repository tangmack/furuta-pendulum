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
import cv2

import math


G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.5  # length of pendulum 2 in m

M1 = 2.5  # mass of pendulum 1 in kg
R1 = 0.8
J1 = 3.0

M2 = 1.0  # mass of pendulum 2 in kg
R2 = 0.75
J2 = 2.0
# FRIC1 = .3  # friction coefficient (* by velocity)
# FRIC2 = .3  # friction coefficient (* by velocity)

A1 = J1 + M1 * R1 * R1
A2 = M1 * R1 * L2
A3 = J2 + M2 * R2 * R2 + M1 * L2 * L2
A4 = M1 * G * R1


def derivs(state, t):

    dydx = np.zeros_like(state)

    phi = A1 * state[3] * state[3] * sin(2*state[1]) \
        + A2 * state[3] * (state[1] + A2 * state[3] * cos(state[0]))*sin(state[0])/A1

    

    dydx[0] = state[1] + A2 * state[3] * cos(state[0])
    dydx[1] = A4 * sin(state[0]) + phi

    dydx[2] = state[3]
    dydx[3] = 0  # Apply a constant torque top motor of 0.1

    print t
    return dydx

# create a time array from 0..100 sampled at 0.01 second steps
dt = 0.03125
t = np.arange(0.0, 7.0, dt)
print "t is"
print t
myfps = int(1/dt)
myinterval = int(dt*1000)

# Initial conditions
q1 = 135.0  # angle of pendulum
q2 = 0.0  # angle of Furuta arm/motor
q1d = 0.0  # initial angular speed of q1
q2d = 1.0  # initial angular speed of q2


# X1, X2, X3, X4 are the state space representation
X1 = q1
X2 = A1 * q1d - A2 * cos(math.radians(q1) * math.radians(q2d))
X3 = q2
X4 = q2d

# initial state
state = np.radians([X1, X2, X3, X4])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = -L1*sin(y[:, 0])
print y[:,0]
y1 = L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2])
y2 = -L2*cos(y[:, 2])

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
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

t1 = time.time()
print(t1-t0)
