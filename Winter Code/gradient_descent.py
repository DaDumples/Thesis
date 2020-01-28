from numpy import *
from numpy.linalg import *
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import ode
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import sys
import datetime
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

from simulate_lightcurve import *
import simulation_config as config


ESTIMATE = config.ESTIMATE()
VARIABLES = config.VARIABLES()

ALBEDO = ESTIMATE.ALBEDO
MEASUREMENT_VARIANCE = ESTIMATE.MEASUREMENT_VARIANCE

ROTATION = VARIABLES.ROTATION
DT = VARIABLES.DT
LAT = VARIABLES.LATITUDE
LON = VARIABLES.LONGITUDE
ALT = VARIABLES.ALTITUDE

SC = RF.Premade_Spacecraft().CYLINDER

lightcurve = load('plate_true_lightcurve.npy')
time = load('plate_time.npy')
OBS_VECS = load('plate_obs_vecs.npy')
attitudes = load('plate_true_attitude.npy')
SUN_VEC = array([0,1,0])


def main():


    #angular_velocity_est = array([.01,.01,.01])
    angular_velocity_actual = array([0, 0.2, 0])
    state_est = array([0.0,0.0,0.0])
    DIM_X = 3
    DIM_Z = 10

    #points = MerweScaledSigmaPoints(DIM_X, alpha = .3, beta = 2, kappa = .1)
    change = 1
    num_iters = 2000
    state_estimates = zeros((num_iters, 3))
    for buh in range(num_iters):

        
        indices = sort(random.randint(0, high = len(lightcurve), size = DIM_Z))
        times = [time[i] for i in indices]
        meas  = [lightcurve[i] for i in indices]

        derivatives = zeros(len(state_est))
        for variable in range(len(state_est)):
            delta = .01

            modified_state = copy(state_est)
            modified_state[variable] += delta
            # quat = modified_state[0:4]
            # quat = quat/norm(quat)
            # modified_state[0:4] = quat
            samples = test_sameple_times(times, hstack([modified_state, angular_velocity_actual]), SC)
            error1 = norm(meas - samples)

            modified_state = copy(state_est)
            modified_state[variable] -= delta
            # quat = modified_state[0:4]
            # quat = quat/norm(quat)
            # modified_state[0:4] = quat
            samples = test_sameple_times(times, hstack([modified_state, angular_velocity_actual]), SC)
            error2 = norm(meas - samples)

            derivatives[variable] = (error1 - error2)/2/delta
            #print(derivatives)
        #print(derivatives[4:7])
            #print(error1, error2)

        #print(derivatives)
        state_est += -derivatives*array([1.0e5,1.0e5,1.0e5])

        #print('[{0:.4f}, {1:.4f}, {2:.4f}] [{3:.4f}, {4:.4f}, {5:.4f}] {6:>4}'.format(*state_est, buh), end = '\r')
        print('[{0:.4f}, {1:.4f}, {2:.4f}]  {3:>4}'.format(*state_est, buh), end = '\r')

        # quat = state_est[0:4]
        # quat = quat/norm(quat)
        # state_est[0:4] = quat

        change = norm(derivatives)
        #print(state_est)
        state_estimates[buh] = state_est
        #print(state_estimates[-1])

    #state_estimates = vstack(state_estimates)
    #print(state_estimates)
    quat_ests = vstack([hstack(CF.dcm2quat(CF.euler2dcm(ROTATION, eulers))) for eulers in state_estimates[:,0:3]])

    attitude_error = attitudes[0][0:4] - quat_ests
    #rate_error  = array([0,.2,0]) - state_estimates[:,3:6]

    save('plate_state_estimates', state_estimates)

    plt.figure()
    plt.plot(attitude_error)
    plt.title('Attitude Error vs Iteration')
    plt.xlabel('Iteration #')
    plt.ylabel('Error')
    plt.savefig('Attitude Error vs Iteration.png', bbox_inches = 'tight')

    # plt.figure()
    # plt.plot(rate_error)
    # plt.title('Rate Error vs Iteration')
    # plt.xlabel('Iteration #')
    # plt.ylabel('Error')
    # plt.savefig('Rate Error vs Iteration.png', bbox_inches = 'tight')

    solver = ode(propagate_eulers)
    solver.set_integrator('dopri5')
    solver.set_initial_value(hstack([state_est, angular_velocity_actual]), 0)
    rotations = []
    for t in time:
        solver.integrate(t)
        rotations.append(solver.y[0:3])

    print('')
    print('AHH',len(rotations), len(time))

    rotations = vstack(rotations)
    est_lightcurve = states_to_lightcurve(time, OBS_VECS, rotations, SC)

    plt.figure()
    plt.plot(time,lightcurve)
    plt.plot(time, est_lightcurve)
    plt.title('True vs Resulting Lightcurve')
    plt.xlabel('Time [s]')
    plt.ylabel('Power [W]')
    plt.savefig('Truth vs Estimated Lightcurve.png', bbox_inches = 'tight')

    plt.show()




def test_sameple_times(times, initial_state, SC):

    solver = ode(propagate_eulers)
    solver.set_integrator('dopri5')
    solver.set_initial_value(initial_state, 0)

    meas_est = zeros(len(times))
    for i, t in enumerate(times):
        solver.integrate(t)
        eulers = solver.y[0:3]
        dcm_body2eci = CF.euler2dcm(ROTATION, eulers)
        obs_vec_body = dcm_body2eci.T@OBS_VECS[i]
        sun_vec_body = dcm_body2eci.T@SUN_VEC
        meas_est[i] = SC.calc_reflected_power(obs_vec_body, sun_vec_body)

    return meas_est

def propagate_eulers(t, state):
    phi = state[0]
    theta = state[1]
    #zeta = state[2]
    omega = state[3:6]

    A = array([[1, sin(phi)*tan(theta)  , cos(phi)*tan(theta) ],
               [0, cos(phi)             , -sin(phi)           ],
               [0, sin(phi)*(1/cos(theta)) , cos(phi)*(1/cos(theta)) ]])

    deulers = A@omega

    domega = zeros(3)

    return hstack([deulers, domega])


def propagate_quats(t, state):
    eta = state[0]
    eps = state[1:4]
    omega = state[4:7]

    deps = .5*(eta*identity(3) + CF.crux(eps))@omega
    deta = -.5*dot(eps, omega)
    domega = zeros(3)

    derivatives = hstack([deta, deps, domega])

    return derivatives


if __name__ == "__main__":

    main()

    

    

