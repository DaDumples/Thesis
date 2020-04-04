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


LAT = 37.1348
LON = -12.2110
ALT = 684

def propagate_orbit(t, state, mu = 398600):
    pos = state[0:3]
    vel = state[3:6]

    d_pos = vel
    d_vel = -pos*mu/norm(pos)**3

    return hstack([d_pos, d_vel])

login_info = sys.argv[1]
Directory = sys.argv[2]
norad_id = sys.argv[2]

login_file = open(login_info, 'r')
username = login_file.readline().replace('\n','')
password = login_file.readline().replace('\n','')
st = SpaceTrackClient(username, password)


for passfile in [os.path.join(Directory,file) for file in os.listdir(Directory) if os.path.isdir(os.path.join(Directory,file))]:
    time.sleep(2)
    print(passfile)
    if (not os.path.join(passfile, 'sunvec.npy') in os.listdir(passfile)) or (not os.path.join(passfile, 'obsvec.npy') in os.listdir(passfile)):

        julian_dates = load(os.path.join(passfile, 'julian_dates0.npy'))

        timesteps = diff(julian_dates)*24*3600
        plt.plot(timesteps)
        plt.xlabel('Index')
        plt.ylabel('Timestep')
        plt.savefig(os.path.join(passfile,'timesteps.png'))
        plt.close()

        sample_times = (julian_dates - julian_dates[0])*24*3600

        dt1 = julian.from_jd(julian_dates[0])
        dt2 = dt1 + dt.timedelta(days = 1)
        drange = op.inclusive_range(dt1, dt2)
        try:
            obj_data = st.tle(epoch = drange, norad_cat_id = norad_id)[0]
        except Exception as e:
            print('Failed to find object ID: '+str(norad_id))
            print(e)
            continue

        line1 = obj_data['TLE_LINE1']
        line2 = obj_data['TLE_LINE2']
        name = obj_data['TLE_LINE0'][2:]

        file = open(os.path.join(passfile,'data0.txt'),'w')
        for key in obj_data.keys():
            file.write(key + ': ' + str(obj_data[key]))
            file.write('\n')
        file.close()

        Satellite = twoline2rv(line1, line2, wgs84)

        date0 = julian.from_jd(julian_dates[0])
        sc_pos, sc_vel = Satellite.propagate(*date0.timetuple()[0:6])
        state0 = hstack([sc_pos, sc_vel])

        solver = ode(propagate_orbit)
        solver.set_integrator('dopri5')
        solver.set_initial_value(state0, 0)
        #times = asarray(arange(0, _pass['Pass Length'], DT))

        positions = []
        times = []
        for t in sample_times:
            solver.set_initial_value(state0, 0)
            solver.integrate(t)
            positions.append(solver.y)
            times.append(solver.t)
        times = hstack(times)
        positions = vstack(positions)

        obs_vecs = []
        sun_vecs = []
        sat_poss = []
        site_poss = []
        range_vecs = []

        
        for t, state in zip(times, positions):
            date = date0 + datetime.timedelta(seconds = t)
            lst = AF.local_sidereal_time(date, LON)
            site = AF.observation_site(LAT, lst, ALT)
            #sc_pos, sc_vel = Satellite.propagate(*date.timetuple()[0:6])
            sc_pos = state[0:3]
            #sc_pos = asarray(sc_pos)
            range_vec = sc_pos - site
            sun_vec = AF.vect_earth_to_sun(date)

            obs_vecs.append(site - sc_pos)
            sun_vecs.append(sun_vec)
            sat_poss.append(sc_pos)
            site_poss.append(site)
            range_vecs.append(range_vec)

        obs_vecs = vstack(obs_vecs)
        sun_vecs = vstack(sun_vecs)
        sat_poss = vstack(sat_poss)
        site_poss = vstack(site_poss)
        range_vecs = vstack(range_vecs)

        save(passfile+'/satpos.npy', sat_poss)
        save(passfile+'/sitepos.npy', site_poss)
        save(passfile+'/rangevec.npy', range_vecs)
        save(passfile+'/sunvec.npy', sun_vecs)
        save(passfile+'/obsvec.npy', obs_vecs)
        save(passfile+'/times.npy', times)