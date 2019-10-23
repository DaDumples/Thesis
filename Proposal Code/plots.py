import matplotlib.pyplot as plt
from numpy import *
from numpy.linalg import *
from simulate_lightcurve import *
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import sys
import os

save = False
tag = ''
for indx, arg in enumerate(sys.argv):
    if arg == '-s':
        save = True
    elif arg == '-tag':
        tag = sys.argv[indx+1]


ROTATION = 'xyz'

now = '_'.join(str(datetime.now()).split())
now = now.replace('.','').replace(':','-')

path = 'plots/'
path_ext = str(datetime.now()).split()[0]

if not path_ext in os.listdir(os.getcwd()+'\\'+path):
    if save:
        os.makedirs(path+path_ext)

path += path_ext + '/'

truth = load('true_states.npy')
time = load('time.npy')
est = load('estimated_states.npy')
true_curve = load('true_lightcurve.npy')

est_curve = states_to_lightcurve(est)





tru_eci_w = []
est_eci_w = []
true_eulers = []
est_quats = []
for t_state, e_state in zip(truth, est):
    t_w = t_state[4:7]
    e_w = e_state[3:6]

    eta = t_state[0]
    eps = t_state[1:4]
    eulers = e_state[0:3]
    truth_body2eci = R.from_quat(hstack([eps, eta])).as_dcm()
    est_body2eci = R.from_euler(ROTATION, eulers).as_dcm()

    tru_eci_w.append(truth_body2eci@t_w)
    est_eci_w.append(est_body2eci@e_w)
    true_eulers.append(R.from_quat(hstack([eps, eta])).as_euler(ROTATION))
    est_quat = R.from_euler(ROTATION, eulers).as_quat()
    est_quats.append(hstack([est_quat[3], est_quat[0:3]]))

tru_eci_w = vstack(tru_eci_w)
est_eci_w = vstack(est_eci_w)
true_eulers = vstack(true_eulers)
est_quats = vstack(est_quats)




plt.figure()
plt.plot(truth[:,0:4],'b')
plt.plot(est_quats,'r')



angle_errs = truth[:,0:4] - est_quats

fig, axes = plt.subplots(nrows = 4, ncols = 1, sharex = True)
axes[0].plot(time, angle_errs[:,0])
axes[1].plot(time, angle_errs[:,1])
axes[2].plot(time, angle_errs[:,2])
axes[3].plot(time, angle_errs[:,3])

axes[3].set_xlabel('Time [s]')
fig.suptitle('Quaternion Error vs Time')

if save:
    plt.savefig(path+now+'Quaternion_Errors_'+tag+'.png', bbox_inches = 'tight')




w_err = truth[:,4:7] - est[:,3:6]

fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True)
axes[0].plot(time, w_err[:,0])
axes[1].plot(time, w_err[:,1])
axes[2].plot(time, w_err[:,2])

axes[2].set_xlabel('Time [s]')
axes[0].set_ylabel('X-rate err [rad/s]')
axes[1].set_ylabel('Y-rate err [rad/s]')
axes[2].set_ylabel('Y-rate err [rad/s]')
fig.suptitle('BODY Angular Velocity Error vs Time')

if save:
    plt.savefig(path+now+'BODY_Angular_Errors_'+tag+'.png', bbox_inches = 'tight')


ang_sign = 1
if norm(tru_eci_w - est_eci_w) > norm(tru_eci_w + est_eci_w):
    print('Angular velocity flipped')
    ang_sign = -1

w_err = tru_eci_w[:,0:3] - ang_sign*est_eci_w[:,0:3]

fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True)
axes[0].plot(time, w_err[:,0])
axes[1].plot(time, w_err[:,1])
axes[2].plot(time, w_err[:,2])

axes[2].set_xlabel('Time [s]')
axes[0].set_ylabel('X-rate err [rad/s]')
axes[1].set_ylabel('Y-rate err [rad/s]')
axes[2].set_ylabel('Y-rate err [rad/s]')
fig.suptitle('ECI Angular Velocity Error vs Time')

if save:
    plt.savefig(path+now+'ECI_Angular_Errors_'+tag+'.png', bbox_inches = 'tight')


fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True)

axes[0].plot(time[-500:], true_curve[-500:])
axes[1].plot(time[-500:], est_curve[-500:])

axes[1].set_xlabel('Time [s]')
axes[0].set_ylabel('"True" Light Curve')
axes[1].set_ylabel('Filtered Light Curve')
fig.suptitle('True vs Filtered Light Curve (Normalized)')

if save:
    plt.savefig(path+now+'Truth_vs_Filtered_'+tag+'.png', bbox_inches = 'tight')


plt.figure()
plt.plot(time, states_to_lightcurve(true_eulers) - est_curve)
plt.title('Light Curve Error')
plt.xlabel("Time [s]")
plt.ylabel("Error")

if save:
    plt.savefig(path+now+'Light_Curve_Error_'+tag+'.png', bbox_inches = 'tight')

# plt.figure()
# plt.plot(time[:2000], true_curve[:2000])
# plt.xlabel('Time [s]')
# plt.ylabel('Normalized Light Curve')
# plt.title('Simulated Light Curve of Tumbling Cube')

# plt.savefig(path+now+'Light_Curve_.png', bbox_inches = 'tight')

plt.show()