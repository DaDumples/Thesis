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

excel_file = sys.argv[1]


#drange = op.inclusive_range(dt1, dt2)
#lines = st.tle_latest(iter_lines = True, epoch = drange, norad_cat_id = 22830, format = 'tle') 

print('Opening file...')
df = pd.read_excel(excel_file)
print('File opened!')

print('Grouping Detection Request IDs...')
pass_numbers = list(set(df['Detection Request Id']))
print('Done grouping.')

pass_dfs = []
for num in pass_numbers:
    print(num, end = '\r')
    sub_df = df[df['Detection Request Id'] == num]
    if len(sub_df['Flux']) > 100:
        pass_dfs.append(sub_df)
print('{} passes found.'.format(len(pass_dfs)))

for p in pass_dfs:
    julian_day = asarray(p['Time Jd Int'])[0]
    julian_day_integers = asarray(p['Time Jd Int'])
    norad_id = asarray(p['Norad Id'])[0]
    error_mean = p['Flux Err'].mean()

    print(norad_id, end = '\r')

    day_fractions = asarray(p['Time Jd Frac'])
    julian_dates = day_fractions + julian_day_integers

    sample_rate = (julian_dates[1] - julian_dates[0])*24*3600
    date0 = julian.from_jd(julian_day + day_fractions[0])

    Directory = str(int(norad_id))
    if not os.path.exists(Directory):
        os.makedirs(Directory)

    pass_start_date = julian.from_jd(julian_dates[0])
    pass_directory = os.path.join(Directory, str(pass_start_date).replace(':','-'))
    if not os.path.exists(pass_directory):
        os.makedirs(pass_directory)

    number = len([x for x in os.listdir(Directory) if 'data' in x ])

    datafilename = os.path.join(pass_directory, 'data{}.txt'.format(number))
    datafile = open(datafilename, 'w')
    datafile.write('Start JD: '+str(julian_dates[0])+'\n')
    datafile.write('Start Date: '+str(date0)+'\n')
    datafile.write('Sample Rate: '+str(sample_rate)+'\n')
    datafile.write('Error Mean: '+str(error_mean)+'\n')
    datafile.close()

    

    save(os.path.join(pass_directory,'lightcurve{}.npy'.format(number)), asarray(p['Flux']))
    save(os.path.join(pass_directory,'julian_dates{}.npy'.format(number)), asarray(julian_dates))

    plt.plot(julian_dates, p['Flux'])
    plt.xlabel('Julian Date')
    plt.ylabel('Flux/Counts')
    plt.savefig(os.path.join(pass_directory,'Lightcurve{}.png'.format(number)), bbox_inches = 'tight')
    plt.close()

