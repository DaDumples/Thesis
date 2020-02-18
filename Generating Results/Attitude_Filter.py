from numpy import *
from numpy.linalg import *
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import ode
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, MMAEFilterBank, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import sys
sys.path.insert(0, '../../../Aero_Funcs')
sys.path.insert(0, '..')
import datetime
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

import Aero_Funcs as AF
import Controls_Funcs as CF
import Aero_Plots as AP
import Reflection_Funcs as RF

import json

def propagate_mrps(t, state, true_inertia):
    mrps = state[0:3]
    omega = state[3:6]

    d_mrps = CF.mrp_derivative(mrps, omega)
    d_omega = inv(true_inertia)@(-cross(omega, true_inertia@omega))

    return hstack([d_mrps, d_omega])

def propagate_mrps_inertia(t, state):
    mrps = state[0:3]
    omega = state[3:6]

    est_inertia = diag([1, abs(state[6]), abs(state[7])])

    d_mrps = CF.mrp_derivative(mrps, omega)
    d_omega = inv(est_inertia)@(-cross(omega, est_inertia@omega))

    return hstack([d_mrps, d_omega, 0, 0])

def modified_rodrigues_prop(state, dt, inertia = None, est_inertia = False):

    #print(state)
    if not est_inertia:
        solver = ode(propagate_mrps)
    else:
        solver = ode(propagate_mrps_inertia)

    solver.set_integrator('dopri5')
    solver.set_initial_value(state, 0)

    if not est_inertia:
        solver.set_f_params(inertia)

    solver.integrate(dt)

    return hstack([solver.y])

def mrp_measurement_function(state, obsvec, sunvec, Geometry):

    mrps = state[0:3]
    dcm_body2eci = CF.mrp2dcm(mrps)

    obs_vec_body = dcm_body2eci.T@obsvec
    sun_vec_body = dcm_body2eci.T@sunvec

    return array([Geometry.calc_reflected_power(obs_vec_body, sun_vec_body)])


def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} Loading:[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')

def Attitude_Filter(lightcurve, obsvecs, sunvecs, dt, noise, Geometry, Inertia_Matrix, est_inertia = False):
    if est_inertia:
        DIM_X = 8
        DIM_Z = 1
        points = JulierSigmaPoints(DIM_X)
        initial_rate = array([.01, .01, .01])
        initial_mrps = array([.01, .01, .01])
        initial_inertia = array([1.0, 1.0])
        initial_state = hstack([initial_mrps, initial_rate, initial_inertia])

        kf = UnscentedKalmanFilter(dim_x = DIM_X, dim_z = DIM_Z, dt = dt,
                                   fx = modified_rodrigues_prop,
                                   hx = mrp_measurement_function,
                                   points = points)
        kf.x = initial_state
        kf.P = diag(ones(DIM_X))*100
        kf.R = noise**2
        kf.Q = diag([1, 1, 1, 1e3, 1e3, 1e3, 1e3, 1e3])*noise*1e-2

    else:
        DIM_X = 6
        DIM_Z = 1
        points = JulierSigmaPoints(DIM_X)
        initial_rate = array([.01, .01, .01])
        initial_mrps = array([.01, .01, .01])

        initial_state = hstack([initial_mrps, initial_rate])

        kf = UnscentedKalmanFilter(dim_x = DIM_X, dim_z = DIM_Z, dt = dt,
                                   fx = modified_rodrigues_prop,
                                   hx = mrp_measurement_function,
                                   points = points)
        kf.x = initial_state
        kf.P = diag(ones(DIM_X))*100
        kf.R = noise**2
        kf.Q = diag([1, 1, 1, 1e3, 1e3, 1e3])*noise*1e-2
        #print(kf.Q)


    noisy_lightcurve = lightcurve + random.normal(0, noise, len(lightcurve))
    # plt.plot(noisy_lightcurve)
    # plt.plot(lightcurve)
    # plt.show()

    means = zeros((len(lightcurve), DIM_X))
    covariances = zeros((len(lightcurve), DIM_X, DIM_X))
    residuals = zeros(len(lightcurve))
    filtered_lightcurve = zeros(len(lightcurve))

    len_residual_buffer = int(10/dt)
    if len(lightcurve) < len_residual_buffer:
        len_residuals = int(len(lightcurve)/10)
    


    RMS = 1000.0
    dRMS = 1.0
    buffnum = 1
    for i, (z, obsvec, sunvec) in enumerate(zip(noisy_lightcurve, obsvecs, sunvecs)):

        kf.predict(dt = dt, inertia = Inertia_Matrix, est_inertia = est_inertia)
        kf.update(z, obsvec = obsvec, sunvec = sunvec, Geometry = Geometry)

        #print('[{0:+.4f}, {1:+.4f}, {2:+.4f}] [{3:+.4f}, {4:+.4f}, {5:+.4f}] {6:<10}'.format(*kf.x, i), end = '\r')


        #switch from mrps to shadow parameters
        if norm(kf.x[0:3]) > 1:
            mrps = kf.x[0:3]
            kf.x[0:3] = -mrps/norm(mrps)

        if norm(kf.x[3:6]) > 1:
            kf.x[3:6] = kf.x[3:6]/norm(kf.x[3:6])

        if est_inertia:
            if abs(kf.x[6]) > 5:
                kf.x[6] = 5

            if abs(kf.x[7]) > 5:
                kf.x[7] = 5

        #print(kf.sigmas_f)
        #print(kf.y)
        loading_bar(i/len(lightcurve), 'Filtering Data')

        means[i, :] = kf.x
        covariances[i, :, :] = kf.P
        residuals[i] = kf.y
        filtered_lightcurve[i] = kf.zp

        # if floor(i/len_residual_buffer) > buffnum:
        #     RMS_n = sqrt(sum(residuals[-len_residual_buffer:]))
        #     print(sum(residuals[-len_residual_buffer:]))
        #     dRMS = (RMS - RMS_n)/RMS
        #     RMS = RMS_n
        #     buffnum += 1

        # if dRMS < .01:
        #     break



    return means, covariances, residuals, filtered_lightcurve