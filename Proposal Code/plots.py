import matplotlib.pyplot as plt
from numpy import *
from numpy.linalg import *
from simulate_lightcurve import *
from datetime import datetime

now = '_'.join(str(datetime.now()).split())
now = now.replace('.','').replace(':','-')

path = 'plots/'

truth = load('true_states.npy')
time = load('time.npy')
est = load('estimated_states.npy')
true_curve = load('true_lightcurve.npy')

est_curve = states_to_lightcurve(est)


fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True)
axes[0].plot(time, truth[:,-3] - est[:,-3])
axes[1].plot(time, truth[:,-2] - est[:,-2])
axes[2].plot(time, truth[:,-1] - est[:,-1])

axes[2].set_xlabel('Time [s]')
axes[0].set_ylabel('X-rate err [rad/s]')
axes[1].set_ylabel('Y-rate err [rad/s]')
axes[2].set_ylabel('Y-rate err [rad/s]')
fig.suptitle('Angular Velocity Error vs Time')

plt.savefig(path+'Angular_Errors_'+now+'.png', bbox_inches = 'tight')


fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True)

axes[0].plot(time[:500], true_curve[:500])
axes[1].plot(time[:500], est_curve[:500])

axes[1].set_xlabel('Time [s]')
axes[0].set_ylabel('"True" Light Curve')
axes[1].set_ylabel('Filtered Light Curve')
fig.suptitle('True vs Filtered Light Curve (Normalized)')

plt.savefig(path+'Truth_vs_Filtered_'+now+'.png', bbox_inches = 'tight')

plt.figure()
plt.plot(time[:2000], true_curve[:2000])
plt.xlabel('Time [s]')
plt.ylabel('Normalized Light Curve')
plt.title('Simulated Light Curve of Tumbling Cube')

plt.savefig(path+'Light_Curve_'+now+'.png', bbox_inches = 'tight')

plt.show()