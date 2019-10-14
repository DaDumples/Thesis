from numpy import *
from numpy.linalg import *
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import ode
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import minimize

from simulate_lightcurve import *

FACETS = [array([ 1, 0, 0]),
              array([-1, 0, 0]),
              array([ 0, 1, 0]),
              array([ 0,-1, 0]),
              array([ 0, 0, 1]),
              array([ 0, 0,-1])]

# INERTIA = array([[1,   .12, .7],
#                  [.12,  2,  .01],
#                  [.7,  .01,   1]])

INERTIA = array([[1,   .02, .5],
                 [.02,  1,  .1],
                 [.5,  .1,   1]])

# INERTIA = array([[1,  0,   0],
#                  [0,  1,   0],
#                  [0,  0,   1]])

ROTATION = 'xyz'


def measurement_function(state):
    eulers = state[:3]
    
    ALBEDO = .6

    AREAS = array([2,2,2,2,1,1])
    OBS_VEC = array([2,1,3])/norm(array([2,1,3]))
    SUN_VEC = array([1,0,0])
    ALPHA = arccos(dot(OBS_VEC, SUN_VEC))

    dcm_body2eci = R.from_euler(ROTATION, eulers).as_dcm().T

    power = 0
    for facet, area in zip(FACETS, AREAS):
        normal = dcm_body2eci@facet
        power += facet_brightness(OBS_VEC, SUN_VEC, ALBEDO, normal, area)

    return array([power])


def state_transition_function(state, dt):

    solver = ode(propagate)
    solver.set_integrator('dopri5')
    solver.set_initial_value(state, 0)
    solver.set_f_params(INERTIA)

    solver.integrate(solver.t + dt)


    return solver.y

def inflate_inertia(inertia_values):
    inertia = zeros((3,3))
    inertia[0] = inertia_values[0:3]
    inertia[:,0] = inertia_values[0:3].T
    inertia[1,1:] = inertia_values[3:5]
    inertia[1:,1] = inertia_values[3:5].T
    inertia[2,2] = inertia_values[5]

    return inertia


def state_transition_function_inertia(state, dt):

    propagated_state = state[:-6]

    inertia_values = state[-6:]
    inertia = zeros((3,3))
    inertia[0] = inertia_values[0:3]
    inertia[:,0] = inertia_values[0:3].T
    inertia[1,1:] = inertia_values[3:5]
    inertia[1:,1] = inertia_values[3:5].T
    inertia[2,2] = inertia_values[5]

    solver = ode(propagate)
    solver.set_integrator('dopri5')
    solver.set_initial_value(propagated_state, 0)
    solver.set_f_params(inertia)

    solver.integrate(solver.t + dt)

    return hstack([solver.y, inertia_values])


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

    

    DIM_X = 6
    DIM_Z = 1
    dt = .1
    points = MerweScaledSigmaPoints(6, alpha = .3, beta = 2, kappa = .1)

    angular_velocity0 = array([0,0,0])
    eulers = array([pi,pi,pi])

    # flat_inertia0 = array([1,0,0,1,0,1])
    # state0 = hstack([eulers, angular_velocity0, flat_inertia0])

    state0 = hstack([eulers, angular_velocity0])

    #state_guess = array([1,0,0,0,0,0,0])

    lightcurve = load('true_lightcurve.npy')
    time = load('time.npy')

    kf = UnscentedKalmanFilter(dim_x = DIM_X, dim_z = DIM_Z, dt = dt,
                                 fx = state_transition_function,
                                 hx = measurement_function,
                                 points = points)

    
    kf.x = state0
    kf.P = .1
    z_std = .001
    kf.R = z_std**2

    Qrate = Q_discrete_white_noise(dim = 3, dt = dt, block_size = 1)
    Qquat = Q_discrete_white_noise(dim = 3, dt = dt, block_size = 1)

    # kf.Q = vstack([hstack([Qquat, zeros((3,3))]),
    #                hstack([zeros((3,3)), Qrate])])


    G = vstack([zeros((3,3)), inv(INERTIA)])
    kf.Q = G@G.T*.00001

    # Q = zeros((12,12))
    # Q[3:6, 3:6]  = Q_discrete_white_noise(dim = 3, dt = dt, block_size = 1)
    # Q[6:, 6:] = diag(random.normal(0,.1, size = (1,6)))

    # kf.Q = Q*.00001

    Xs, Ps = kf.batch_filter(lightcurve)

    # Xs = []
    # num_pts = len(lightcurve)
    # percentage = 10
    # for i, z in enumerate(lightcurve):
    #     kf.predict()
    #     kf.update(z)
    #     Xs.append(kf.x)

    #     if i*100/num_pts > percentage:
    #         print('%',percentage)
    #         percentage += 10

    # Xs = vstack(Xs)



    save('estimated_states', Xs)
    save('estimated_covariances', Ps)

    est_curve = states_to_lightcurve(Xs)

    plt.figure()
    plt.plot(time, est_curve)
    plt.plot(time, lightcurve)

    plt.figure()
    truth = load('true_states.npy')
    error = norm(truth[:,-3:] - Xs[:,-3:], axis = 1)
    plt.plot(error)

    plt.show()

