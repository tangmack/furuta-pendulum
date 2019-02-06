import time
t0 = time.time()

import odrive
from odrive.enums import *

print("finding an odrive...")
my_drive = odrive.find_any()

# Calibrate motor and wait for it to finish
print("starting calibration...")
my_drive.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
while my_drive.axis0.current_state != AXIS_STATE_IDLE:
    time.sleep(0.1)

time.sleep(1)

# To read a value, simply read the property
print("Bus voltage is " + str(my_drive.vbus_voltage) + "V")

my_drive.axis0.controller.config.vel_limit = 70000
print("Velocity limit is ", str(my_drive.axis0.controller.config.vel_limit))
# my_drive.axis0.motor.config.current_lim = 30
my_drive.axis0.controller.config.control_mode = CTRL_MODE_CURRENT_CONTROL
print("Ctrl mode is ", str(my_drive.axis0.controller.config.control_mode))

my_drive.axis0.controller.current_setpoint = 0.3
print("Current setpoint is ", str(my_drive.axis0.controller.current_setpoint))

my_drive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
print("State: ", str(my_drive.axis0.current_state))

while my_drive.axis0.current_state != AXIS_STATE_IDLE:
    print(my_drive.vbus_voltage)
    print(my_drive.axis0.motor.current_control.Iq_measured)
    time.sleep(0.1)

# time.sleep(40)
#
# while my_drive.axis0.requested_state == AXIS_STATE_CLOSED_LOOP_CONTROL:
#     print(my_drive.vbus_voltage)
#     time.sleep(0.1)
# print(hex(mot/error)

my_drive.axis0.requested_state = AXIS_STATE_IDLE

t1 = time.time()
print("code run time: ",t1-t0)