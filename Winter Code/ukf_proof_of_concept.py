from numpy import *
from numpy.linalg import *
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import ode
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, MMAEFilterBank
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import sys
import datetime
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

from simulate_lightcurve import *
import simulation_config as config
import Aero_Funcs as AF
import Aero_Plots as AP

ESTIMATE = config.ESTIMATE()
VARIABLES = config.VARIABLES()

ALBEDO = ESTIMATE.ALBEDO
INERTIA = ESTIMATE.INERTIA
MEASUREMENT_VARIANCE = ESTIMATE.MEASUREMENT_VARIANCE

ROTATION = VARIABLES.ROTATION
DT = VARIABLES.DT
LAT = VARIABLES.LATITUDE
LON = VARIABLES.LONGITUDE
ALT = VARIABLES.ALTITUDE
PASS = VARIABLES.PASS

# SPACECRAFT = twoline2rv(PASS['TLE'][0], PASS['TLE'][1], wgs84)
# UTC0 = PASS['EPOCH']
# TELESCOPE_ECEF = AF.lla_to_ecef(LAT, LON, ALT, geodetic = True)
SC = RF.Premade_Spacecraft().CYLINDER

LIGHTCURVE = load('plate_true_lightcurve.npy')
TIME = load('plate_time.npy')
OBS_VECS = load('plate_obs_vecs.npy')
#SUN_VECS = load('sun_vecs.npy')
SUN_VEC = array([0,1,0])


def main():

    DIM_X = 6
    DIM_Z = 1
    points = MerweScaledSigmaPoints(DIM_X, alpha = .3, beta = 2, kappa = .1)

    angular_velocity0 = zeros(3) #random.rand(3)/10
    eulers = zeros(3) #random.rand(3)*2*pi
    state0 = hstack([eulers, angular_velocity0])
    
    kf = UnscentedKalmanFilter(dim_x = DIM_X, dim_z = DIM_Z, dt = DT,
                                 fx = no_change,
                                 hx = test_sameple_times,
                                 points = points)
    kf.x = state0
    kf.P = diag([.2**2, .2**2, .2**2, 1e-12, 1e-12, 1e-12])
    R = .3**2
    kf.R = R

    #When filtering inertia
    #Q = zeros((DIM_X,DIM_X))
    #Q = Q_discrete_white_noise(dim = 1, dt = DT, block_size = 6)
    Q = diag([(2e-4)**2, (2e-4)**2, (2e-4)**2, (10e-12)**2, (10e-12)**2, (10e-6)**2])
    kf.Q = Q*.001

    print(Q)

    means = zeros((len(LIGHTCURVE), DIM_X))
    # state covariances from Kalman Filter
    covariances = zeros((len(LIGHTCURVE), DIM_X, DIM_X))

    noisy_lightcurve = LIGHTCURVE + random.normal(0, MEASUREMENT_VARIANCE, len(LIGHTCURVE))

    # num_iters = len(LIGHTCURVE)
    # for i, (z, t) in enumerate(zip(noisy_lightcurve, TIME)):
    #     kf.predict(dt=DT)
    #     kf.update(z, index = i)

    for i in range(1000):

        indices = sort(random.randint(0, high = len(LIGHTCURVE), size = DIM_Z))
        times = [TIME[i] for i in indices]
        meas  = [noisy_lightcurve[i] for i in indices]

        kf.predict(dt = DT)
        kf.update(meas, times = times, SC = SC)


        means[i, :] = kf.x
        covariances[i, :, :] = kf.P
        print('[{0:.4f}, {1:.4f}, {2:.4f}] [{3:.4f}, {4:.4f}, {5:.4f}] {6:>4}'.format(*kf.x, i), end = '\r')

    (Xs, Ps, Ks) = kf.rts_smoother(means, covariances)

    save('estimated_states', Xs)
    save('estimated_covariances', Ps)

    plt.plot(LIGHTCURVE)
    plt.plot(states_to_lightcurve(TIME, OBS_VECS, Xs[0:3], SC))
    plt.show()

    plt.savefig('Truth vs Recreated Lightcurve')


def no_change(state, dt):
    return state

def state_transition_function_const_omega(state, dt):

    #print(state)
    # solver = ode(propagate_eulers_only)
    # solver.set_integrator('lsoda')
    # solver.set_initial_value(state, 0)

    # for i in range(10):
    #     solver.integrate(solver.t + dt/10)

    #return hstack([solver.y])

    return propagate_eulers_only(0, state)*dt + state


def propagate_eulers_only(t, state):
    phi = state[0]
    theta = state[1]
    omega = state[3:]


    A = array([[1, sin(phi)*tan(theta)  , cos(phi)*tan(theta) ],
               [0, cos(phi)             , -sin(phi)           ],
               [0, sin(phi)*(1/cos(theta)) , cos(phi)*(1/cos(theta)) ]])

    deulers = A@omega
    return hstack([deulers, 0,0,0])

def measurement_function(state, index):
    eulers = state[0:3]
    omega = state[3:6]
    dcm_body2eci = CF.euler2dcm(ROTATION, eulers)

    obs_vec_body = dcm_body2eci.T@OBS_VECS[index]
    sun_vec_body = dcm_body2eci.T@SUN_VEC

    return array([SC.calc_reflected_power(obs_vec_body, sun_vec_body)])

def test_sameple_times(initial_state, times, SC):

    solver = ode(propagate_eulers_only)
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


def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} Loading:[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')



if __name__ == "__main__":

    main()

    

    

