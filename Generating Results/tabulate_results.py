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
import Controls_Funcs as CF

import json
import pandas as pd


def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} :[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')

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

pass_directories = [file for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]

data = {}
columns = ['X Rate Error', 'X STD', 'X % Error', 'Y Rate Error', 'Y STD', 'Y % Error', 'Z Rate Error', 'Z STD', 'Z % Error',
           'OBS X Rate Error', 'OBS X STD', 'OBS X % Error', 'OBS Y Rate Error', 'OBS Y STD', 'OBS Y % Error', 'OBS Z Rate Error', 'OBS Z STD', 'OBS Z % Error']
for i, filename in enumerate(pass_directories):
    passfile = os.path.join(Directory, filename)
    loading_bar(i/len(pass_directories), text = 'Processing')
    for resultfile in [file for file in os.listdir(passfile) if os.path.isdir(os.path.join(passfile,file))]:



        run_number = resultfile[7]
        result_dir = os.path.join(passfile,resultfile)
        inertia_filtered = 'inertia' in resultfile

        rowname = filename + '_' + str(run_number)
        data[rowname] = []

        times = load(os.path.join(passfile, 'time.npy'))

        true_lightcurve = load(os.path.join(passfile, 'lightcurve'+run_number+'.npy'))
        true_mrps = load(os.path.join(passfile, 'mrps'+run_number+'.npy'))
        true_rates = load(os.path.join(passfile, 'angular_rate'+run_number+'.npy'))

        obs_vecs = load(os.path.join(passfile, 'obsvec.npy'))
        sun_vecs = load(os.path.join(passfile, 'sunvec.npy'))

        est_lightcurve = load(os.path.join(result_dir, 'results'+run_number+'_estimated_curve.npy'))
        means = load(os.path.join(result_dir, 'results'+run_number+'_raw_means.npy'))
        covariances = load(os.path.join(result_dir, 'results'+run_number+'_raw_covariance.npy'))
        residuals = load(os.path.join(result_dir, 'results'+run_number+'_raw_residuals.npy'))

        eci_rate_true = vstack(list([CF.mrp2dcm(true_mrps[0])@true_rates[0] for m, rate in zip(true_mrps, true_rates)]))
        eci_rate_ests = []
        for est, tru in zip(means, eci_rate_true):
            eci_rate_est = CF.mrp2dcm(est[0:3])@est[3:6]
            a = norm(eci_rate_est - tru)
            b = norm(-eci_rate_est - tru)
            if b < a:
                eci_rate_ests.append(-eci_rate_est)
            else:
                eci_rate_ests.append(eci_rate_est)
        eci_rate_ests = vstack(eci_rate_ests)


        obs_frame_true_rate = []
        obs_frame_est_rate = []
        for obs, sun, truth, est in zip(obs_vecs, sun_vecs, eci_rate_true, eci_rate_ests):

            x = obs/norm(obs)
            z = cross(sun, obs)/norm(cross(sun,obs))
            y = cross(z, x)

            eci2obs = vstack([x, y, z])

            obs_frame_true_rate.append(eci2obs@truth)
            obs_frame_est_rate.append(eci2obs@est)


        obs_frame_true_rate = vstack(obs_frame_true_rate)
        obs_frame_est_rate = vstack(obs_frame_est_rate)


        mrp_std = []
        rate_std = []
        inertia_std = []
        for cov in covariances:
            mrp_vals, mrp_vecs = eig(cov[0:3, 0:3])
            mrp_std.append(abs(sum(sqrt(mrp_vals)*mrp_vecs, axis = 1)))

            rate_vals, rate_vecs = eig(cov[3:6,3:6])
            rate_std.append(abs(sum(sqrt(rate_vals)*rate_vecs, axis = 1)))

            if inertia_filtered:
                inertia_vals, inertia_vecs = eig(cov[6:8,6:8])
                inertia_std.append(abs(sum(sqrt(inertia_vals)*inertia_vecs, axis = 1)))
        mrp_std = vstack(mrp_std)
        rate_std = vstack(rate_std)
        if inertia_filtered:
            inertia_std = vstack(inertia_std)

        for i, (truth, _mean) in enumerate(zip(true_rates, means[:,3:6])):
            if dot(truth, _mean) < 0:
                means[i, 3:6] = -_mean

        for i, truth in enumerate(true_mrps):
            if norm(truth) > 1:
                true_mrps[i] = -truth/norm(truth)

        final_errors = true_rates[-12:-2, :] - means[-12:-2, 3:6]
        X_err = mean(final_errors[:,0])
        data[rowname].append(X_err)
        X_std = std(final_errors[:,0])
        data[rowname].append(X_std)
        X_percent_err = X_err/mean(true_rates[-12:-2,0])*100
        data[rowname].append(X_percent_err)

        Y_err = mean(final_errors[:,1])
        data[rowname].append(Y_err)
        Y_std = std(final_errors[:,1])
        data[rowname].append(Y_std)
        Y_percent_err = Y_err/mean(true_rates[-12:-2,1])*100
        data[rowname].append(Y_percent_err)

        Z_err = mean(final_errors[:,2])
        data[rowname].append(Z_err)
        Z_std = std(final_errors[:,2])
        data[rowname].append(Z_std)
        Z_percent_err = Z_err/mean(true_rates[-12:-2,2])*100
        data[rowname].append(Z_percent_err)


        final_errors = obs_frame_true_rate[-12:-2, :] - obs_frame_est_rate[-12:-2, :]
        X_err = mean(final_errors[:,0])
        data[rowname].append(X_err)
        X_std = std(final_errors[:,0])
        data[rowname].append(X_std)
        X_percent_err = X_err/mean(true_rates[-12:-2,0])*100
        data[rowname].append(X_percent_err)

        Y_err = mean(final_errors[:,1])
        data[rowname].append(Y_err)
        Y_std = std(final_errors[:,1])
        data[rowname].append(Y_std)
        Y_percent_err = Y_err/mean(true_rates[-12:-2,1])*100
        data[rowname].append(Y_percent_err)

        Z_err = mean(final_errors[:,2])
        data[rowname].append(Z_err)
        Z_std = std(final_errors[:,2])
        data[rowname].append(Z_std)
        Z_percent_err = Z_err/mean(true_rates[-12:-2,2])*100
        data[rowname].append(Z_percent_err)

df = pd.DataFrame.from_dict(data, orient = 'index', columns = columns)
df.to_csv(os.path.join(Directory, 'datatable.csv'))
df.to_pickle(os.path.join(Directory, 'datatable.pkl'))




