import numpy as np
from twojointarm_funs import *

# ============================================================================

A1 = np.array([55,90]) * np.pi/180   # 55,90 degrees shoudler,elbow
H1,E1 = joints_to_hand(A1, aparams)
H2 = H1 + np.array([0, 0.15])        # 15 cm movement distance
mt = 0.400                           # movement time (sec)
sr = 1000                            # sample rate (Hz)
npts = np.int(mt*sr)+1               # number of time points

# get a minimum-jerk desired hand trajectory
t,H,Hd,Hdd = minjerk(H1,H2,mt,npts)

# get corresponding desired joint angles velocities and accelerations
A,Ad,Add = hand_to_joints((H,Hd,Hdd),aparams)

# compute required joint torques
Q = inverse_dynamics(A,Ad,Add,aparams)

# run a forward simulation using those joint torques Q
A0, Ad0 = A[0,:], Ad[0,:] # starting joint angles and velocities
A_sim, Ad_sim, Add_sim = forward_dynamics(A0, Ad0, Q, t, aparams)

# make some plots
import matplotlib.pyplot as plt
%matplotlib

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



# ============================================================================
# Forward simulations with random amplification or diminishment of 
# shoulder and elbow torques

# re-draw simulated movement from above

f1 = plt.figure()
f1ax1 = f1.add_subplot(2,2,1)
lines = f1ax1.plot(t,A*180/np.pi)
f1ax1.set_xlabel('TIME (sec)')
f1ax1.set_ylabel('JOINT ANGLES (deg)')
f1ax2 = f1.add_subplot(2,2,2)
lines = f1ax2.plot(t,Ad*180/np.pi)
f1ax2.set_xlabel('TIME (sec)')
f1ax2.set_ylabel('JOINT VELOCITIES (deg/s)')
f1ax3 = f1.add_subplot(2,2,3)
lines = f1ax3.plot(t,Add*180/np.pi)
f1ax3.set_xlabel('TIME (sec)')
f1ax3.set_ylabel('JOINT ACCELERATIONS (deg/s)')
f1ax4 = f1.add_subplot(2,2,4)
lines = f1ax4.plot(t,Q)
f1ax4.set_xlabel('TIME (sec)')
f1ax4.set_ylabel('JOINT TORQUES (Nm)')
f1.tight_layout()

H,_ = joints_to_hand(A, aparams)

f2 = plt.figure()
f2ax1 = f2.add_subplot(1,2,1)
f2ax2 = f2.add_subplot(1,2,2)
f2ax1.plot(H[0,0],H[0,1],'r.')
f2ax2.plot(H[-1,0],H[-1,1],'rs')
f2ax1.set_xlabel('X (m)')
f2ax2.set_xlabel('X (m)')
f2ax1.set_ylabel('Y (m)')
f2ax2.set_ylabel('Y (m)')
f2.tight_layout()


# re-simulate forward simulation by amplifying or diminishing joint torques
# within a given range, and replot

n_perts = 100

for i in range(n_perts):
	QQ = np.copy(Q)
	QQ[:,0] = QQ[:,0] * np.random.uniform(0.90, 1.10)
	QQ[:,1] = QQ[:,1] * np.random.uniform(0.90, 1.10)
	A0, Ad0 = A[0,:], Ad[0,:] # starting joint angles and velocities
	A_sim, Ad_sim, Add_sim = forward_dynamics(A0, Ad0, QQ, t, aparams)
	f1ax1.plot(t,A_sim[:,0]*180/np.pi,'g-')
	f1ax1.plot(t,A_sim[:,1]*180/np.pi,'r-')
	f1ax2.plot(t,Ad_sim[:,0]*180/np.pi,'g-')
	f1ax2.plot(t,Ad_sim[:,1]*180/np.pi,'r-')
	f1ax3.plot(t,Add_sim[:,0]*180/np.pi,'g-')
	f1ax3.plot(t,Add_sim[:,1]*180/np.pi,'r-')
	f1ax4.plot(t,QQ[:,0],'g-')
	f1ax4.plot(t,QQ[:,1],'r-')
	H_sim,_ = joints_to_hand(A_sim, aparams)
	f2ax1.plot(H_sim[:,0],H_sim[:,1],'b-')
	f2ax2.plot(H_sim[-1,0],H_sim[-1,1],'b.')
f2ax2.plot(H[-1,0],H[-1,1],'rs')
f2ax1.axis('equal')
f2ax2.axis('equal')

