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
import Aero_Funcs as AF
import Aero_Plots as AP

ESTIMATE = config.ESTIMATE()
VARIABLES = config.VARIABLES()

FACETS = ESTIMATE.FACETS
ALBEDO = ESTIMATE.ALBEDO
AREAS = ESTIMATE.AREAS
OBS_VEC = ESTIMATE.OBS_VEC
SUN_VEC = ESTIMATE.SUN_VEC
INERTIA = ESTIMATE.INERTIA
MEASUREMENT_VARIANCE = ESTIMATE.MEASUREMENT_VARIANCE

ROTATION = VARIABLES.ROTATION
DT = VARIABLES.DT
LAT = VARIABLES.LATITUDE
LON = VARIABLES.LONGITUDE
ALT = VARIABLES.ALTITUDE
PASS = VARIABLES.PASS

SPACECRAFT = twoline2rv(PASS['TLE'][0], PASS['TLE'][1], wgs84)
UTC0 = PASS['EPOCH']
TELESCOPE_ECEF = AF.lla_to_ecef(LAT, LON, ALT, geodetic = True)


def main():

    case = 0
    for indx, arg in enumerate(sys.argv):
        if arg == '--2inertia':
            case = 1
            print('Including Inertia into state')
        elif arg == '--fullinertia':
            case = 2
        elif arg == '--inertialoffset':
            case = 3
            print('Estimating inertial offset')
        elif arg == '--constomega':
            case = 4
            print('Assuming constant omega')
        elif arg == 'ukf_proof_of_concept.py':
            pass
        else:
            print('Unknown Argument')


    angular_velocity0 = array([.1,.1,.1])
    eulers = array([0,0,0])

    if case == 1:
        DIM_X = 8
        stf = state_transition_function_inertia
        flat_inertia0 = array([1,1])
        state0 = hstack([eulers, angular_velocity0, flat_inertia0])
    elif case == 0:
        DIM_X = 6
        stf = state_transition_function
        state0 = hstack([eulers, angular_velocity0])
    elif case == 2:
        DIM_X = 11
        stf = state_transition_function_full_inertia
        flat_inertia0 = array([0,0,1,0,1])
        state0 = hstack([eulers, angular_velocity0, flat_inertia0])
    elif case == 3:
        DIM_X = 11
        stf = state_transition_function_eigens_and_eulers
        flat_inertia0 = array([1,1])
        euler_offsets = array([0,0,0])
        state0 = hstack([eulers, angular_velocity0, flat_inertia0, euler_offsets])
    elif case == 4:
        DIM_X = 6
        stf = state_transition_function_const_omega
        state0 = hstack([eulers, angular_velocity0])


    
    DIM_Z = 1
    points = MerweScaledSigmaPoints(DIM_X, alpha = .3, beta = 2, kappa = .1)


    lightcurve = load('true_lightcurve.npy')
    reverse_lightcurve = flip(lightcurve.copy())
    time = load('time.npy')

    kf = UnscentedKalmanFilter(dim_x = DIM_X, dim_z = DIM_Z, dt = DT,
                                 fx = stf,
                                 hx = measurement_function,
                                 points = points)
    kf.x = state0
    kf.P = 1
    z_std = MEASUREMENT_VARIANCE
    R = z_std**2
    kf.R = R

    #When filtering inertia
    Q = zeros((DIM_X,DIM_X))
    Q[3:6, 3:6]  = Q_discrete_white_noise(dim = 3, dt = DT, block_size = 1)
    kf.Q = Q*.00001

    means = zeros((len(lightcurve), DIM_X))
    # state covariances from Kalman Filter
    covariances = zeros((len(lightcurve), DIM_X, DIM_X))

    for i, (z, t) in enumerate(zip(lightcurve, time)):
        kf.predict(dt=DT)
        kf.update(z, R, time = t)
        means[i, :] = kf.x
        covariances[i, :, :] = kf.P


    (Xs, Ps, Ks) = kf.rts_smoother(means, covariances)

    save('estimated_states', Xs)
    save('estimated_covariances', Ps)

    plt.plot(lightcurve)
    plt.plot(states_to_lightcurve(Xs))
    plt.show()

def propagate_eulers_only(t, state):
    phi = state[0]
    theta = state[1]
    omega = state[3:]


    A = array([[1, sin(phi)*tan(theta)  , cos(phi)*tan(theta) ],
               [0, cos(phi)             , -sin(phi)           ],
               [0, sin(phi)*(1/cos(theta)) , cos(phi)*(1/cos(theta)) ]])

    deulers = A@omega
    return hstack([deulers, 0,0,0])

def measurement_function(state, time):

    print(state)
    eulers = state[:3]
    dcm_body2eci = R.from_euler(ROTATION, eulers).as_dcm().T
    power = 0

    now = UTC0 + datetime.timedelta(seconds = time)
    sc_eci, _ = SPACECRAFT.propagate(now.year, now.month, now.day, now.hour, now.minute, now.second)
    sc_eci = asarray(sc_eci)

    lst = AF.local_sidereal_time(now, LON)
    telescope_eci = AF.Cz(lst)@TELESCOPE_ECEF       
    obs_vec = telescope_eci - sc_eci

    sun_vec = AF.vect_earth_to_sun(now)

    for facet, area in zip(FACETS, AREAS):
        normal = dcm_body2eci@facet
        power += phong_brdf(OBS_VEC, SUN_VEC, normal, area)

    return array([power])


def state_transition_function(state, dt):

    solver = ode(propagate)
    solver.set_integrator('lsoda', atol = 1e-8, rtol = 1e-8)
    solver.set_initial_value(state, 0)
    solver.set_f_params(INERTIA)

    for i in range(10):
        solver.integrate(solver.t + dt/10)

    return solver.y

def reverse_state_transition_function(t, state, inertia):
    return -propagate(t, state, inertia)
    

def inflate_inertia(inertia_values):
    inertia = zeros((3,3))
    inertia[0] = inertia_values[0:3]
    inertia[:,0] = inertia_values[0:3].T
    inertia[1,1:] = inertia_values[3:5]
    inertia[1:,1] = inertia_values[3:5].T
    inertia[2,2] = inertia_values[5]

    return inertia


def state_transition_function_inertia(state, dt):

    #print(state)
    propagated_state = state[0:6]

    inertia_values = state[6:]
    inertia = diag(hstack([1,inertia_values]))
    solver = ode(propagate)
    solver.set_integrator('lsoda')
    solver.set_initial_value(propagated_state, 0)
    solver.set_f_params(inertia)

    for i in range(10):
        solver.integrate(solver.t + dt/10)

    return hstack([solver.y, inertia_values])

def state_transition_function_const_omega(state, dt):

    #print(state)
    solver = ode(propagate_eulers_only)
    solver.set_integrator('lsoda')
    solver.set_initial_value(state, 0)

    for i in range(10):
        solver.integrate(solver.t + dt/10)

    return hstack([solver.y])

def state_transition_function_full_inertia(state, dt):

    #print(state)

    propagated_state = state[0:6]

    inertia_values = state[6:]
    inertia = inflate_inertia(hstack([1, inertia_values]))
    solver = ode(propagate)
    solver.set_integrator('lsoda')
    solver.set_initial_value(propagated_state, 0)
    solver.set_f_params(inertia)

    for i in range(10):
        solver.integrate(solver.t + dt/10)

    return hstack([solver.y, inertia_values])

def state_transition_function_eigens_and_eulers(state, dt):

    #print(state)

    propagated_state = state[0:6]
    inertia_values = state[6:8]
    offset_eulers = state[8:11]

    inertial_offset = R.from_euler(ROTATION,offset_eulers).as_dcm()

    inertia = inertial_offset@diag(hstack([1, inertia_values]))@inertial_offset.T

    solver = ode(propagate)
    solver.set_integrator('lsoda')
    solver.set_initial_value(propagated_state, 0)
    solver.set_f_params(inertia)

    for i in range(10):
        solver.integrate(solver.t + dt/10)

    return hstack([solver.y, inertia_values, offset_eulers])

def reverse_state_transition_function_inertia(state, dt):
    return -state_transition_function_inertia(state, dt)


def linear_state_transition_function(state, dt):
    for i in range(10):
        state += propagate(0,state, INERTIA)*dt
    return state

def minmize_quaternion(q_guess, q_samples, weights):
    eta_guess = q_guess[0]
    eps_guess = q_guess[1:4]
    Aq = quat2dcm(eps_guess, eta_guess)

    summ = 0
    for quat, weight in zip(q_samples, weights):
        eta = quat[0]
        eps = quat[1:4]
        Aqi = quat2dcm(eps, eta)
        summ += weight*norm(Aq - Aqi, ord = 'fro')

    return summ



def quaternion_mean_function(sigmas, weights):

    q_guess = sigmas[0, 0:4]
    q_samples = sigmas[:,0:4]
    q_ave = minimize(minmize_quaternion, q_guess, args = (q_samples, weights), tol = 1e-4).x

    omegas = sigmas[:,4:]
    ave_omega = zeros(3)
    for omega, weight in zip(omegas, weights):
        ave_omega += omega*weight/len(weights)

    return hstack([q_ave, ave_omega])

def radian_residual(x, y):
    z = x - y

    for ang in z[:3]:
        ang = ang%(2*pi)

    return z


if __name__ == "__main__":

    main()

    

    

