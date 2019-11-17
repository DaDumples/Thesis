from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
from scipy.spatial.transform import Rotation as R
import pymap3d as pm
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import datetime
import sys


sys.path.insert(0, '../../Aero_Funcs')

import read_lightcurve_data as rlc
import simulation_config as config
import Aero_Funcs as AF
import Aero_Plots as AP



TRUTH = config.TRUTH()
VARIABLES = config.VARIABLES()

FACETS = TRUTH.FACETS
ALBEDO = TRUTH.ALBEDO
AREAS = TRUTH.AREAS
OBS_VEC = TRUTH.OBS_VEC
SUN_VEC = TRUTH.SUN_VEC
INERTIA = TRUTH.INERTIA
MEASUREMENT_VARIANCE = TRUTH.MEASUREMENT_VARIANCE

ROTATION = VARIABLES.ROTATION
DT = VARIABLES.DT
PASS = VARIABLES.PASS
LAT = VARIABLES.LATITUDE
LON = VARIABLES.LONGITUDE
ALT = VARIABLES.ALTITUDE

def main():

    angular_velocity0 = array([0.1,0.1,.1])
    eta0 = 1
    eps0 = array([0,0,0])

    eulers = array([0,0,0])

    state0 = hstack([eta0, eps0, angular_velocity0])

    solver = ode(propagate_quats)
    solver.set_integrator('lsoda')
    solver.set_initial_value(state0, 0)
    solver.set_f_params(INERTIA, False)

    newstate = []
    time = []

    tspan = PASS['TIME'][-1]

    while solver.successful() and solver.t < tspan:

        newstate.append(solver.y)
        time.append(solver.t)

        solver.integrate(solver.t + DT)

    newstate = vstack(newstate)
    time = hstack(time)


    #Generate lightcurve

    lightcurve = states_to_lightcurve(time, newstate, PASS, quats = True)



    #lightcurve += random.normal(0, MEASUREMENT_VARIANCE, size = lightcurve.shape)


    #Save Data

    save('true_lightcurve', lightcurve)
    save('true_states', newstate)
    save('time', time)

    sc_pos, tel_pos, sun_pos = get_positions(time, PASS)

    colors = []
    for x, y in zip(sc_pos, sun_pos):
        if AF.shadow(x, y) == 0:
            colors.append('k')
        else:
            colors.append('b')


    fig = plt.figure()
    ax = Axes3D(fig)
    AP.plot_earth(ax)
    ax.scatter(sc_pos[:,0], sc_pos[:,1], sc_pos[:,2], c = colors)
    AP.plot_orbit(ax, tel_pos)
    AP.scale_plot(ax)


    azimuth, elevation = AF.pass_az_el(tel_pos, sc_pos)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = 'polar')
    #matplotlib takes azimuth in radians and elevation in degrees for some reason :/
    ax.scatter(radians(azimuth), 90-elevation, c = colors)
    ax.set_ylim([0,90])



    plt.figure()
    plt.plot(time, lightcurve)
    plt.show()



def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1],A[0], 0]])

def quat2dcm(E,n):
    #n = scalar part
    #E = vector part
    
    dcm = (2*n**2 - 1)*identity(3) + 2*outer(E,E) - 2*n*crux(E) 

    return dcm


def propagate_quats(t, state, inertia, noise = False):
    eta = state[0]
    eps = state[1:4]
    omega = state[4:]

    deps = .5*(eta*identity(3) + crux(eps))@omega
    deta = -.5*dot(eps, omega)
    domega = -inv(inertia)@crux(omega)@inertia@omega

    if noise:
        domega += random.normal(0, .000001)

    derivatives = hstack([deta, deps, domega])

    return derivatives

def propagate(t, state, inertia, noise = False):
    phi = state[0]
    theta = state[1]
    #zeta = state[2]
    omega = state[3:6]

    A = array([[1, sin(phi)*tan(theta)  , cos(phi)*tan(theta) ],
               [0, cos(phi)             , -sin(phi)           ],
               [0, sin(phi)*(1/cos(theta)) , cos(phi)*(1/cos(theta)) ]])

    deulers = A@omega

    domega = -inv(inertia)@crux(omega)@inertia@omega
    if noise:
        domega += random.normal(0, .000001)

    return hstack([deulers, domega])

def states_to_lightcurve(times, states, pass_obj, quats = False):

    lightcurve = []
    spacecraft = twoline2rv(pass_obj['TLE'][0], pass_obj['TLE'][1], wgs84)
    telescope_ecef = AF.lla_to_ecef(LAT, LON, ALT, geodetic = True)
    utc0 = pass_obj['EPOCH']

    sun_vec = AF.vect_earth_to_sun(utc0)

    for time, state in zip(times, states):

        now = utc0 + datetime.timedelta(seconds = time)

        sc_eci, _ = spacecraft.propagate(now.year, now.month, now.day, now.hour, now.minute, now.second)
        sc_eci = asarray(sc_eci)

        # if AF.shadow(sc_eci,sun_vec):
        #     #if the spacecraft is in shadow its brightness is zero
        #     print("IN SHADOW")
        #     lightcurve.append(0)
        #     continue

        lst = AF.local_sidereal_time(now, LON)
        
        telescope_eci = AF.Cz(lst)@telescope_ecef        
        obs_vec = telescope_eci - sc_eci

        if quats:
            eta = state[0]
            eps = state[1:4]
            dcm_body2eci = R.from_quat(hstack([eps, eta])).as_dcm().T
        else:
            eulers = state[:3]
            dcm_body2eci = R.from_euler(ROTATION, eulers).as_dcm().T

        power = 0
        for facet, area in zip(FACETS, AREAS):
            normal = dcm_body2eci@facet
            power += facet_brightness(obs_vec, sun_vec, ALBEDO, normal, area)

        lightcurve.append(power)

    lightcurve = hstack(lightcurve)

    return lightcurve


def get_positions(times, pass_obj):

    spacecraft = twoline2rv(pass_obj['TLE'][0], pass_obj['TLE'][1], wgs84)
    telescope_ecef = AF.lla_to_ecef(LAT, LON, ALT, geodetic = True)
    utc0 = pass_obj['EPOCH']

    telescope_ecis = []
    spacecraft_ecis = []
    sun_ecis = []

    for time in times:

        now = utc0 + datetime.timedelta(seconds = time)

        sun_vec = AF.vect_earth_to_sun(now)

        sc_eci, _ = spacecraft.propagate(now.year, now.month, now.day, now.hour, now.minute, now.second)
        sc_eci = asarray(sc_eci)

        lst = AF.local_sidereal_time(now, LON)
        
        telescope_eci = AF.Cz(lst)@telescope_ecef        
        obs_vec = telescope_eci - sc_eci

        telescope_ecis.append(telescope_eci)
        spacecraft_ecis.append(sc_eci)
        sun_ecis.append(sun_vec)

    return vstack(spacecraft_ecis), vstack(telescope_ecis), vstack(sun_ecis)


def facet_brightness(obs_vec, sun_vec, albedo, normal, area):
    #determines the brightness of a facet at 1 meter distance
    #obs_dot is the dot product of the facet normal and observation vector
    #sun_dot is the dot product of the facet normal and sun vector
    #solar_phase_angle is the angle between the sun and observation vector
    #albedo is albedo
    #area is the facet area
    #from LIGHTCURVE INVERSION FOR SHAPE ESTIMATION OF GEO OBJECTS FROM
    #SPACE-BASED SENSORS

    obs_norm = obs_vec/norm(obs_vec)
    sun_norm = sun_vec/norm(sun_vec)

    obs_dot = dot(normal, obs_norm)
    sun_dot = dot(normal, sun_norm)

    if obs_dot <= 0 or sun_dot <= 0:
        return 0

    solar_phase_angle = arccos(dot(obs_norm, sun_norm))

    #constants from above paper
    c = .1
    A0 = .5
    D = .1
    k = -.5


    phase = A0*exp(-solar_phase_angle/D) + k*solar_phase_angle + 1

    scattering = phase*obs_dot*sun_dot*(1/(obs_dot + sun_dot) + c)
    brightness = scattering*albedo*area


    return brightness

if __name__ == '__main__':
    main()


