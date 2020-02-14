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

from simulate_lightcurve import *
import simulation_config as config
import Aero_Funcs as AF
import Aero_Plots as AP

import json

#Luca was here
def propagate_mrps(t, state):
    mrps = state[0:3]
    omega = state[3:6]
    inertia = diag([1, abs(state[6]), abs(state[7])])

    #dcm_eci2body = CF.mrp2dcm(mrps).T
    d_mrps = CF.mrp_derivative(mrps, omega)
    d_omega = inv(inertia)@(-cross(omega, inertia@omega))

    #print(hstack([d_mrps, d_omega, 0, 0]))

    return hstack([d_mrps, d_omega, 0, 0])

def modified_rodrigues_prop(state, dt):

    #print(state)
    solver = ode(propagate_mrps)
    solver.set_integrator('dopri5')
    solver.set_initial_value(state, 0)

    solver.integrate(dt)

    return hstack([solver.y])

def modified_rodrigues_const_prop(state, dt):

    #print(state)
    solver = ode(propagate_mrps_only)
    solver.set_integrator('dopri5')
    solver.set_initial_value(state, 0)

    solver.integrate(dt)

    return hstack([solver.y])

def propagate_mrps_only(t, state):
    mrps = state[0:3]
    omega = state[3:6]

    #dcm_eci2body = CF.mrp2dcm(mrps).T
    d_mrps = CF.mrp_derivative(mrps, omega)


    return hstack([d_mrps, 0, 0, 0])

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


def Attitude_Filter(lightcurve, obsvecs, sunvecs, dt, noise, Geometry, inertia = False):
    if inertia:
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
        kf.Q = diag([1, 1, 1, 1e3, 1e3, 1e3, 1e3, 1e3])*10**(-noise-2)

    else:
        DIM_X = 6
        DIM_Z = 1
        points = JulierSigmaPoints(DIM_X)
        initial_rate = array([.01, .01, .01])
        initial_mrps = array([.01, .01, .01])

        initial_state = hstack([initial_mrps, initial_rate])

        kf = UnscentedKalmanFilter(dim_x = DIM_X, dim_z = DIM_Z, dt = dt,
                                   fx = modified_rodrigues_const_prop,
                                   hx = mrp_measurement_function,
                                   points = points)
        kf.x = initial_state
        kf.P = diag(ones(DIM_X))*100
        kf.R = noise**2
        kf.Q = diag([1, 1, 1, 1e3, 1e3, 1e3])*10**(-noise-2)


    noisy_lightcurve = lightcurve + random.normal(0, noise, len(lightcurve))
    plt.plot(noisy_lightcurve)
    plt.plot(lightcurve)
    plt.show()

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

        kf.predict(dt = DT)
        kf.update(z, obsvec = obsvec, sunvec = sunvec, Geometry = Geometry)

        #switch from mrps to shadow parameters
        if norm(kf.x[0:3]) > 1:
            mrps = kf.x[0:3]
            kf.x[0:3] = -mrps/norm(mrps)

        if norm(kf.x[3:6]) > 1:
            kf.x[3:6] = kf.x[3:6]/norm(kf.x[3:6])

        if est_inertia_flag:
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




Simulation_Configuration = json.load(open(sys.argv[1], 'r'))
Satellite = twoline2rv(Simulation_Configuration['TLE Line1'],
                       Simulation_Configuration['TLE Line2'], wgs84)
Lat = Simulation_Configuration['Observation Site Lattitude']
Lon = Simulation_Configuration['Observation Site East Longitude']
Alt = Simulation_Configuration['Observation Site Altitude']
DT = Simulation_Configuration['Data Rate']
Inertia = asarray(Simulation_Configuration['Inertia Matrix'])
Geometry = RF.Premade_Spacecraft().get_geometry(Simulation_Configuration['Spacecraft Geometry'])
Noise_STD = Simulation_Configuration['Sensor STD']
Directory = Simulation_Configuration['Directory']

est_inertia_flag = False
if '-i' in sys.argv:
    est_inertia_flag = True

for passfile in [os.path.join(Directory,file) for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]:
    print(passfile)
    obsvecs = load(os.path.join(passfile, 'obsvec.npy'))
    sunvecs = load(os.path.join(passfile, 'sunvec.npy'))
    curvenum = 0
    filename = lambda name, n: '{0}{1}.npy'.format(name, n)
    while filename('lightcurve',curvenum) in os.listdir(passfile):
        lightcurve = load(os.path.join(passfile, filename('lightcurve', curvenum)))

        means, covariances, residuals, filtered_lightcurve = Attitude_Filter(lightcurve, obsvecs, sunvecs, DT, Noise_STD, Geometry, inertia = est_inertia_flag)

        save(filename+'_raw_means', means)
        save(filename+'_raw_covariance', covariances)
        save(filename+'_raw_residuals', residuals)
        save(filename+'_estimated_curve', filtered_lightcurve)

        curvenum += 1

        try:
            fig, (ax1, ax2) = plt.subplots(2,1,sharex = True)
            ax1.plot(residuals)
            ax1.set_title('Residuals')
            ax2.plot(lightcurve)
            ax2.plot(filtered_lightcurve)
            ax2.set_title('True vs Filtered Lightcurve')
            ax2.legend(['Truth', 'Estimate'])
            plt.savefig('dataset'+curvenum+'.png', bbox_inches = 'tight', dpi = 300)
        except:
            print('Predictably, you fucked up the plots')