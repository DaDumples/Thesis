from numpy import *
from numpy.linalg import *

from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import datetime
import sys, os

sys.path.insert(0, '../../Aero_Funcs')

import Aero_Funcs as AF
import Aero_Plots as AP
import Controls_Funcs as CF
import Reflection_Funcs as RF

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio
import json

def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} :[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')


Simulation_Configuration = json.load(open(sys.argv[1], 'r'))
Geometry = RF.Premade_Spacecraft().get_geometry(Simulation_Configuration['Spacecraft Geometry'])

truth_dir = 'LONG_RECTANGLE_MEO/2020-10-11 19-44-04.719359'
result_dir = 'LONG_RECTANGLE_MEO/2020-10-11 19-44-04.719359/results2'
result_num = result_dir[-1]


obs_vecs = load(truth_dir+'/obsvec.npy')[::50]
sun_vecs = load(truth_dir+'/sunvec.npy')[::50]
attitudes = load(truth_dir+'/mrps'+result_num+'.npy')[::50]

num_frames = len(obs_vecs)

frames = []
count = 0
max_val = 0
for mrps, obs_vec, sun_vec in zip(attitudes, obs_vecs, sun_vecs):
    dcm_eci2body = CF.mrp2dcm(mrps).T
    #dcm_eci2body = CF.mrp2dcm(mrps).T
    image = RF.generate_image(Geometry, dcm_eci2body@obs_vec, dcm_eci2body@sun_vec, win_dim = (6,6), dpm = 20)
    im_max = amax(image)
    if im_max > max_val:
        max_val = im_max
    frames.append(image)
    count += 1
    loading_bar(count/num_frames, 'Rendering gif')

frames = [frame/max_val for frame in frames]

imageio.mimsave(result_dir+'/true_rotation.gif',frames, fps = 10)


attitudes = load(result_dir+'/results'+result_num+'_raw_means.npy')[:,0:3][::50]

frames = []
count = 0
max_val = 0
for mrps, obs_vec, sun_vec in zip(attitudes, obs_vecs, sun_vecs):
    dcm_eci2body = CF.mrp2dcm(mrps).T
    #dcm_eci2body = CF.mrp2dcm(mrps).T
    image = RF.generate_image(Geometry, dcm_eci2body@obs_vec, dcm_eci2body@sun_vec, win_dim = (6,6), dpm = 20)
    im_max = amax(image)
    if im_max > max_val:
        max_val = im_max
    frames.append(image)
    count += 1
    loading_bar(count/num_frames, 'Rendering gif')

frames = [frame/max_val for frame in frames]

imageio.mimsave(result_dir+'/estimated_rotation.gif',frames, fps = 10)