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


est_inertia_flag = False
if '-i' in sys.argv:
    est_inertia_flag = True

pass_directories = [os.path.join(Directory,file) for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]

for i, passfile in enumerate(pass_directories):
    loading_bar(i/len(pass_directories), text = 'Plotting')
    for resultfile in [file for file in os.listdir(passfile) if os.path.isdir(os.path.join(passfile,file))]:

        

        run_number = resultfile[7]
        result_dir = os.path.join(passfile,resultfile)
        inertia_filtered = 'inertia' in resultfile

        times = load(os.path.join(passfile, 'times.npy'))

        true_lightcurve = load(os.path.join(passfile, 'lightcurve'+run_number+'.npy'))
        true_mrps = load(os.path.join(passfile, 'mrps'+run_number+'.npy'))
        true_rates = load(os.path.join(passfile, 'angular_rate'+run_number+'.npy'))

        obs_vecs = load(os.path.join(passfile, 'obsvec.npy'))
        sun_vecs = load(os.path.join(passfile, 'sunvec.npy'))

        est_lightcurve = load(os.path.join(result_dir, 'results'+run_number+'_estimated_curve.npy'))
        means = load(os.path.join(result_dir, 'results'+run_number+'_raw_means.npy'))
        covariances = load(os.path.join(result_dir, 'results'+run_number+'_raw_covariance.npy'))
        residuals = load(os.path.join(result_dir, 'results'+run_number+'_raw_residuals.npy'))


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

        for i, (truth, mean) in enumerate(zip(true_rates, means[:,3:6])):
            if dot(truth, mean) < 0:
                means[i, 3:6] = -mean

        for i, truth in enumerate(true_mrps):
            if norm(truth) > 1:
                true_mrps[i] = -truth/norm(truth)

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        rate_error = true_rates[0:len(times),:] - means[:,3:6]
        ax1.plot(times[::100], rate_error[:,0][::100])
        ax1.plot(times[::100], 3*rate_std[:,0][::100], 'k', alpha = .4)
        ax1.plot(times[::100], -3*rate_std[:,0][::100], 'k', alpha = .4)
        ax1.set_ylim(-.4, .4)
        ax1.set_yticks(arange(-.4, .4, .2))
        ax1.grid()
        ax1.set_title('X-Rate Error')
        ax1.legend(['Error',r'3$\sigma$ Error'])

        ax2.plot(times[::100], rate_error[:,1][::100])
        ax2.plot(times[::100], 3*rate_std[:,1][::100], 'k', alpha = .4)
        ax2.plot(times[::100], -3*rate_std[:,1][::100], 'k', alpha = .4)
        ax2.set_ylim(-.4, .4)
        ax2.set_yticks(arange(-.4, .4, .2))
        ax2.grid()
        ax2.set_title('Y-Rate Error')

        ax3.plot(times[::100], rate_error[:,2][::100])
        ax3.plot(times[::100], 3*rate_std[:,2][::100], 'k', alpha = .4)
        ax3.plot(times[::100], -3*rate_std[:,2][::100], 'k', alpha = .4)
        
        ax3.set_ylim(-.4, .4)
        ax3.set_yticks(arange(-.4, .4, .2))
        ax3.grid()
        ax3.set_title('Z-Rate Error')

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel('Error')

        plt.savefig(os.path.join(result_dir,'Rate Errors.png'),dpi = 300, bbox_inches = 'tight')
        plt.close()

        

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times[::100], true_rates[0:len(times),0][::100])
        ax1.plot(times[::100], means[:,3][::100])

        ax2.plot(times[::100], true_rates[0:len(times),1][::100])
        ax2.plot(times[::100], means[:,4][::100])

        ax3.plot(times[::100], true_rates[0:len(times),2][::100])
        ax3.plot(times[::100], means[:,5][::100])

        ax1.set_title('X-rate Estimate vs Truth')
        ax1.legend(['Truth', 'Est.'])
        ax2.set_title('Y-rate Estimate vs Truth')
        ax3.set_title('Z-rate Estimate vs Truth')

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel('Angular Rate [rad/s]')

        plt.savefig(os.path.join(result_dir,'Rate Comparison.png'),dpi = 300, bbox_inches = 'tight')
        plt.close()


        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        mrp_error = true_mrps[0:len(times),:] - means[:,0:3]
        ax1.plot(times[::100], mrp_error[:,0][::100])
        ax1.plot(times[::100], 3*mrp_std[:,0][::100], 'k', alpha = .4)
        ax1.plot(times[::100], -3*mrp_std[:,0][::100], 'k', alpha = .4)
        ax1.set_ylim(-.4, .4)
        ax1.set_yticks(arange(-.4, .4, .2))
        ax1.grid()
        ax1.set_title('MRP-1 Error')
        ax1.legend(['Error',r'3$\sigma$ Error'])

        ax2.plot(times[::100], mrp_error[:,1][::100])
        ax2.plot(times[::100], 3*mrp_std[:,1][::100], 'k', alpha = .4)
        ax2.plot(times[::100], -3*mrp_std[:,1][::100], 'k', alpha = .4)
        ax2.set_ylim(-.4, .4)
        ax2.set_yticks(arange(-.4, .4, .2))
        ax2.grid()
        ax2.set_title('MRP-2 Error')

        ax3.plot(times[::100], mrp_error[:,2][::100])
        ax3.plot(times[::100], 3*mrp_std[:,2][::100], 'k', alpha = .4)
        ax3.plot(times[::100], -3*mrp_std[:,2][::100], 'k', alpha = .4)
        ax3.set_ylim(-.4, .4)
        ax3.set_yticks(arange(-.4, .4, .2))
        ax3.grid()
        ax3.set_title('MRP-3 Error')

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel('Error')

        plt.savefig(os.path.join(result_dir,'MRP Errors.png'),dpi = 300, bbox_inches = 'tight')
        plt.close()

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times[::100], true_mrps[0:len(times),0][::100])
        ax1.plot(times[::100], means[:,0][::100])

        ax2.plot(times[::100], true_mrps[0:len(times),1][::100])
        ax2.plot(times[::100], means[:,1][::100])

        ax3.plot(times[::100], true_mrps[0:len(times),2][::100])
        ax3.plot(times[::100], means[:,2][::100])

        ax1.set_title('MRP-1 Estimate vs Truth')
        ax1.legend(['Truth', 'Est.'])
        ax2.set_title('MRP-2 Estimate vs Truth')
        ax3.set_title('MRP-3 Estimate vs Truth')

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel('MRP')

        plt.savefig(os.path.join(result_dir,'MRP Comparison.png'),dpi = 300, bbox_inches = 'tight')
        plt.close()

        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
        ax1.plot(times[::100], true_lightcurve[::100])
        ax2.plot(times[::100], est_lightcurve[::100])
        ax3.plot(times[::100], residuals[::100])
        ax1.set_title('True Lightcurve')
        ax2.set_title('Estimated Lightcurve')
        ax3.set_title('Resudial Error')

        ax3.set_xlabel('Time [s]')
        ax2.set_ylabel(r'$W/m^{2}$')

        plt.savefig(os.path.join(result_dir,'LightCurve Residual.png'),dpi = 300, bbox_inches = 'tight')
        plt.close()

        plt.figure()
        angles = zeros(len(sun_vecs))
        for i, (sv, ov) in enumerate(zip(sun_vecs, obs_vecs)):
            angles[i] = degrees(arccos(dot(sv, ov)/norm(sv)/norm(ov)))

        plt.plot(times, angles)
        plt.xlabel('Time [s]')
        plt.ylabel('SPA [Deg]')
        plt.title('Solar Phase Angle During Pass')
        plt.savefig(os.path.join(result_dir,'SPA.png'), bbox_inches = 'tight', dpi = 300)
        plt.close()
