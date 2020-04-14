from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
import sys, os
import julian
import time
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
from scipy.integrate import ode, odeint
import datetime

sys.path.insert(0, '../../Aero_Funcs')

import Aero_Funcs as AF
import Aero_Plots as AP
import Controls_Funcs as CF
import Reflection_Funcs as RF

Directory = sys.argv[1]

def show_date(lc, time):

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(time, lc)
    ax.set_ylabel('Counts')
    ax.set_xlabel('Time [s]')
    ax.set_title('Select the left and right bounds of the data you want to keep. Middle mouse to skip')
    plt.draw()



for passfile in [os.path.join(Directory,file) for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]:

    lightcurve = load(os.path.join(passfile, 'lightcurve0.npy'))
    julian_dates = load(os.path.join(passfile, 'julian_dates0.npy'))
    time = load(os.path.join(passfile, 'time.npy'))
    sunvec = load('sunvec.npy')
    obsvec = load('obsvec.npy')

    time = (julian_dates - julian_dates[0])*24*3600

    #print(time)

    pts = []
    show_date(lightcurve, time)
    pts = asarray(plt.ginput(2, timeout = 60))
    plt.close()
    print(pts)

    if len(pts) == 2:
        idx1 = abs(time - pts[0,0]).argmin()
        idx2 = abs(time - pts[1,0]).argmin()

        orig_data_dir  = os.path.join(passfile,'original_data')
        if not os.path.exists(orig_data_dir):
            os.makedirs(orig_data_dir)
            save(os.path.join(orig_data_dir, 'sunvec.npy'), sunvec)
            save(os.path.join(orig_data_dir, 'obsvec.npy'), obsvec)
            save(os.path.join(orig_data_dir, 'lightcurve0.npy'), lightcurve)
            save(os.path.join(orig_data_dir, 'julian_dates0.npy'), julian_dates)
            save(os.path.join(orig_data_dir, 'time.npy'), time)


        

        

        new_sunvec = sunvec[idx1: idx2]
        new_obsvec = obsvec[idx1: idx2]
        new_lightcurve = lightcurve[idx1: idx2]
        new_julians = julian_dates[idx1: idx2]
        new_time = time[idx1: idx2]

        print(idx1, idx2)
        print(len(new_sunvec), len(new_obsvec), len(new_lightcurve), len(new_julians))


        save(os.path.join(passfile, 'sunvec.npy'), new_sunvec)
        save(os.path.join(passfile, 'obsvec.npy'), new_obsvec)
        save(os.path.join(passfile, 'lightcurve0.npy'), new_lightcurve)
        save(os.path.join(passfile, 'julian_dates0.npy'), new_julians)
        save(os.path.join(passfile, 'time.npy'), new_time)

        plt.plot(new_time, new_lightcurve)
        plt.xlabel('Seconds from pass start')
        plt.ylabel('Flux/Counts')
        plt.savefig(os.path.join(passfile,'Lightcurve0.png'), bbox_inches = 'tight')
        plt.close()





    



