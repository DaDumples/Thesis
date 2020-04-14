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

#Luca was here


Simulation_Configuration = json.load(open(sys.argv[1], 'r'))
Lat = Simulation_Configuration['Observation Site Lattitude']
Lon = Simulation_Configuration['Observation Site East Longitude']
Alt = Simulation_Configuration['Observation Site Altitude']
Inertia = asarray(Simulation_Configuration['Inertia Matrix'])
Geometry = RF.Premade_Spacecraft().get_geometry(Simulation_Configuration['Spacecraft Geometry'])
Noise_STD = Simulation_Configuration['Sensor STD']
Directory = Simulation_Configuration['Directory']
Exposure_Time = Simulation_Configuration['Exposure Time']
Real_Data = Simulation_Configuration['Real Data']


est_inertia_flag = False
if '-i' in sys.argv:
    est_inertia_flag = True

for passfile in [os.path.join(Directory,file) for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]:
    print(passfile)
    obsvecs = load(os.path.join(passfile, 'obsvec.npy'))
    sunvecs = load(os.path.join(passfile, 'sunvec.npy'))
    time = load(os.path.join(passfile, 'time.npy'))

    lightcurves = [filename for filename in os.listdir(passfile) if 'lightcurve' in filename]

    for filename in lightcurves:
        dataname = 'results'+filename.split('.')[0][-1]
        savedir = os.path.join(passfile, dataname)
        if not os.path.exists(savedir):
            os.makedirs(savedir)


            
        lightcurve = load(os.path.join(passfile, filename))

        if not Real_Data:
            lightcurve = lightcurve + random.normal(0, noise, len(lightcurve))

        means, covariances, residuals, filtered_lightcurve = Attitude_Filter(lightcurve, obsvecs, sunvecs, time, Noise_STD, Geometry, Inertia, Exposure_Time, est_inertia_flag)

        

        save(os.path.join(savedir,dataname+'_raw_means'), means)
        save(os.path.join(savedir,dataname+'_raw_covariance'), covariances)
        save(os.path.join(savedir,dataname+'_raw_residuals'), residuals)
        save(os.path.join(savedir,dataname+'_estimated_curve'), filtered_lightcurve)
        if not Real_Data:
            save(os.path.join(savedir,dataname+'_noisy_lightcurve'), lightcurve)

        fig, (ax1, ax2) = plt.subplots(2,1,sharex = True)
        ax1.plot(time, residuals)
        ax1.set_title('Residuals')
        ax2.plot(time, lightcurve)
        ax2.plot(time, filtered_lightcurve)
        ax2.set_xlabel('Time since pass start [s]')
        ax2.set_ylabel('Counts')
        ax2.set_title('True vs Filtered Lightcurve')
        ax2.legend(['Truth', 'Estimate'])
        plt.savefig(os.path.join(savedir,'Lightcurve Comparison.png'), bbox_inches = 'tight', dpi = 300)
        plt.close()