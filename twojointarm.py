import numpy as np
from twojointarm_funs import *

# ============================================================================

A1 = np.array([55,90]) * np.pi/180   # 55,90 degrees shoudler,elbow
H1,E1 = joints_to_hand(A1, aparams)
H2 = H1 + np.array([0, 0.15])        # 15 cm movement distance
mt = 0.400                           # movement time (sec)
sr = 1000                            # sample rate (Hz)

# get a minimum-jerk desired hand trajectory
t,H,Hd,Hdd = minjerk(H1,H2,mt,sr)

# get corresponding desired joint angles velocities and accelerations
A,Ad,Add = hand_to_joints((H,Hd,Hdd),aparams)

# compute required joint torques
Q = inverse_dynamics(A,Ad,Add,aparams)

# run a forward simulation using those joint torques Q
A0, Ad0 = A[0,:], Ad[0,:] # starting joint angles and velocities
A_sim, Ad_sim, Add_sim = forward_dynamics(A0, Ad0, Q, t, aparams)

# make some plots
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(2,2,1)
lines = ax.plot(t,A*180/np.pi)
lines2= ax.plot(t,A_sim*180/np.pi)
ax.set_xlabel('TIME (sec)')
ax.set_ylabel('JOINT ANGLES (deg)')
ax = fig.add_subplot(2,2,2)
lines = ax.plot(t,Ad*180/np.pi)
ax.set_xlabel('TIME (sec)')
ax.set_ylabel('JOINT VELOCITIES (deg/s)')
ax = fig.add_subplot(2,2,3)
lines = ax.plot(t,Add*180/np.pi)
ax.set_xlabel('TIME (sec)')
ax.set_ylabel('JOINT ACCELERATIONS (deg/s)')
ax = fig.add_subplot(2,2,4)
lines = ax.plot(t,Q)
ax.set_xlabel('TIME (sec)')
ax.set_ylabel('JOINT TORQUES (Nm)')
fig.tight_layout()
