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
from scipy.integrate import ode
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
excel_file = sys.argv[2]

login_file = open(login_info, 'r')
username = login_file.readline().replace('\n','')
password = login_file.readline().replace('\n','')
st = SpaceTrackClient(username, password)
#drange = op.inclusive_range(dt1, dt2)
#lines = st.tle_latest(iter_lines = True, epoch = drange, norad_cat_id = 22830, format = 'tle') 

df = pd.read_excel(excel_file)

pass_numbers = list(set(df['Detection Request Id']))

pass_dfs = []
for num in pass_numbers:
    sub_df = df[df['Detection Request Id'] == num]
    if len(sub_df['Flux']) > 100:
        pass_dfs.append(sub_df)

object_occurence = {}
for p in pass_dfs[:10]:
    time.sleep(1)
    julian_day = asarray(p['Time Jd Int'])[0]
    norad_id = asarray(p['Norad Id'])[0]
    error_mean = p['Flux Err'].mean()

    print(norad_id)

    day_fractions = asarray(p['Time Jd Frac'])
    julian_dates = day_fractions + julian_day

    sample_times = (day_fractions - day_fractions[0])*24*3600
    date0 = julian.from_jd(julian_day + day_fractions[0])

    dt1 = julian.from_jd(julian_day)
    dt2 = dt1 + dt.timedelta(days = 1)
    drange = op.inclusive_range(dt1, dt2)
    try:
        obj_data = st.tle(epoch = drange, norad_cat_id = norad_id)[0]
    except:
        print('Failed to find object ID: '+str(norad_id))
        continue
    else:
    # obj_data = st.tle(epoch = drange, norad_cat_id = norad_id)[0]
        line1 = obj_data['TLE_LINE1']
        line2 = obj_data['TLE_LINE2']
        name = obj_data['TLE_LINE0'][2:]

        Satellite = twoline2rv(line1, line2, wgs84)

        if not name in object_occurence:
            object_occurence[name] = 0

        Directory = name.replace('/','')
        if not os.path.exists(Directory):
            os.makedirs(Directory)

        pass_start_date = julian.from_jd(julian_dates[0])
        pass_directory = os.path.join(Directory, str(pass_start_date).replace(':','-'))
        if not os.path.exists(pass_directory):
            os.makedirs(pass_directory)

        sc_pos, sc_vel = Satellite.propagate(*pass_start_date.timetuple()[0:6])
        state0 = hstack([sc_pos, sc_vel])

        solver = ode(propagate_orbit)
        solver.set_integrator('dopri5')
        solver.set_initial_value(state0, 0)
        #times = asarray(arange(0, _pass['Pass Length'], DT))

        positions = []
        times = []
        for t in sample_times:
            if solver.successful():
                positions.append(solver.integrate(t))
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

        number = len([x for x in os.listdir(Directory) if 'data' in x ])

        datafilename = os.path.join(pass_directory, 'data{}.txt'.format(number))
        datafile = open(datafilename, 'w')
        for key in obj_data.keys():
            datafile.write(key +': '+ str(obj_data[key]) +'\n')
        datafile.write('Error Mean: '+str(error_mean))
        datafile.close()

        save(pass_directory+'/satpos.npy', sat_poss)
        save(pass_directory+'/sitepos.npy', site_poss)
        save(pass_directory+'/rangevec.npy', range_vecs)
        save(pass_directory+'/sunvec.npy', sun_vecs)
        save(pass_directory+'/obsvec.npy', obs_vecs)
        save(pass_directory+'/times.npy', times)

        save(os.path.join(pass_directory,'lightcurve{}.npy'.format(number)), asarray(p['Flux']))
        save(os.path.join(pass_directory,'julian_dates{}.npy'.format(number)), asarray(julian_dates))

        plt.plot(julian_dates, p['Flux'])
        plt.savefig(os.path.join(pass_directory,'Lightcurve{}.png'.format(number)), bbox_inches = 'tight')
        plt.close()

