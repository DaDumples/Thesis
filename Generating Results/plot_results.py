from numpy import *
from numpy.linalg import *
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import ode
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, MMAEFilterBank, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import sys, os
sys.path.insert(0, '../../Aero_Funcs')
sys.path.insert(0, '..')
import datetime
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv

import Aero_Funcs as AF
import Aero_Plots as AP
from Attitude_Filter import Attitude_Filter
import Reflection_Funcs as RF

import json

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
    for resultfile in [file for file in os.listdir(passfile) if os.path.isdir(os.path.join(passfile,file))]:

        run_number = resultfile[7]
        result_dir = os.path.join(passfile,resultfile)
        inertia_filtered = 'inertia' in resultfile

        times = load(os.path.join(passfile, 'times.npy'))

        true_lightcurve = load(os.path.join(passfile, 'lightcurve'+run_number+'.npy'))
        true_mrps = load(os.path.join(passfile, 'mrps'+run_number+'.npy'))
        true_rates = load(os.path.join(passfile, 'angular_rate'+run_number+'.npy'))

        est_lightcurve = load(os.path.join(result_dir, 'results'+run_number+'_estimated_curve.npy'))
        means = load(os.path.join(result_dir, 'results'+run_number+'_raw_means.npy'))
        covariances = load(os.path.join(result_dir, 'results'+run_number+'_raw_covariance.npy'))
        residuals = load(os.path.join(result_dir, 'results'+run_number+'_raw_residuals.npy'))


        mrp_std = []
        rate_std = []
        inertia_std = []
        for cov in covariances:
            mrp_vals, mrp_vecs = eig(cov[0:3, 0:3])
            mrp_std.append(sqrt(abs(sum(mrp_vals*mrp_vecs, axis = 1))))

            rate_vals, rate_vecs = eig(cov[3:6,3:6])
            rate_std.append(sqrt(abs(sum(rate_vals*rate_vecs, axis = 1))))

            if inertia_filtered:
                inertia_vals, inertia_vecs = eig(cov[6:8,6:8])
                inertia_std.append(sqrt(abs(sum(inertia_vals*inertia_vecs, axis = 1))))
        mrp_std = vstack(mrp_std)
        rate_std = vstack(rate_std)
        inertia_std = vstack(inertia_std)

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        rate_error = abs(true_rates[0:len(times),:] - means[:,3:6])
        ax1.plot(times, rate_error[:,0])
        ax1.plot(times, rate_std[:,0], 'k')
        ax1.plot(times, -rate_std[:,0], 'k')

        ax2.plot(times, rate_error[:,1])
        ax2.plot(times, rate_std[:,1], 'k')
        ax2.plot(times, -rate_std[:,1], 'k')

        ax3.plot(times, rate_error[:,2])
        ax3.plot(times, rate_std[:,2], 'k')
        ax3.plot(times, -rate_std[:,2], 'k')
        

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times, true_rates[:,0])
        ax1.plot(times, means[:,3])

        ax2.plot(times, true_rates[:,1])
        ax2.plot(times, means[:,4])

        ax3.plot(times, true_rates[:,2])
        ax3.plot(times, means[:,5])

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times, true_mrps[:,0])
        ax1.plot(times, means[:,0])

        ax2.plot(times, true_mrps[:,1])
        ax2.plot(times, means[:,1])

        ax3.plot(times, true_mrps[:,2])
        ax3.plot(times, means[:,2])

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times, true_lightcurve)
        ax2.plot(times, est_lightcurve)
        ax3.semilogy(times, residuals)

        plt.show()
