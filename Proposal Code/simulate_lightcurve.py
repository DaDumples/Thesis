from numpy import *
from numpy.linalg import *
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.spatial.transform import Rotation as R

def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1],A[0], 0]])

def quat2dcm(E,n):
    #n = scalar part
    #E = vector part
    
    dcm = (2*n**2 - 1)*identity(3) + 2*outer(E,E) - 2*n*crux(E) 

    return dcm


# def propagate(t, state, inertia):
#     eta = state[0]
#     eps = state[1:4]
#     omega = state[4:]

#     deps = .5*(eta*identity(3) + crux(eps))@omega
#     deta = -.5*dot(eps, omega)
#     domega = -inv(inertia)@crux(omega)@inertia@omega + random.normal(0, .00001)

#     return hstack([deta, deps, domega])

def propagate(t, state, inertia):
    phi = state[0]
    theta = state[1]
    #zeta = state[2]
    omega = state[3:]

    A = array([[1, sin(phi)*tan(theta)  , cos(phi)*tan(theta) ],
               [0, cos(phi)             , -sin(phi)           ],
               [0, -sin(phi)*1/cos(theta) , cos(phi)*1/cos(theta) ]])

    deulers = A@omega

    domega = -inv(inertia)@crux(omega)@inertia@omega + random.normal(0, .00001)

    return hstack([deulers, domega])


def facet_brightness(obs_vec, sun_vec, albedo, normal, area):
    #determines the brightness of a facet at 1 meter distance
    #obs_dot is the dot product of the facet normal and observation vector
    #sun_dot is the dot product of the facet normal and sun vector
    #solar_phase_angle is the angle between the sun and observation vector
    #albedo is albedo
    #area is the facet area
    #from LIGHTCURVE INVERSION FOR SHAPE ESTIMATION OF GEO OBJECTS FROM
    #SPACE-BASED SENSORS

    obs_dot = dot(normal, obs_vec)
    sun_dot = dot(normal, sun_vec)
    solar_phase_angle = dot(obs_vec, sun_vec)

    #constants from above paper
    c = .1
    A0 = .5
    D = radians(.1)
    k = -.5


    phase = A0*exp(-solar_phase_angle/D) + k*solar_phase_angle + 1
    if phase < 0:
        return 0
    scattering = phase*obs_dot*sun_dot*(1/(obs_dot + sun_dot) + c)
    brightness = scattering*albedo*area
    return brightness

if __name__ == '__main__':


    #facets of a cube
    FACETS = [array([ 1, 0, 0]),
              array([-1, 0, 0]),
              array([ 0, 1, 0]),
              array([ 0,-1, 0]),
              array([ 0, 0, 1]),
              array([ 0, 0,-1])]


    ALBEDO = .6

    AREAS = array([2,2,2,2,1,1])
    OBS_VEC = array([2,1,3])/norm(array([2,1,3]))
    SUN_VEC = array([1,0,0])
    ALPHA = arccos(dot(OBS_VEC, SUN_VEC))

    ROTATION = 'zyx'

    #Propagate rotation

    INERTIA = array([[1,   .02, .5],
                     [.02,  1,  .1],
                     [.5,  .1,   1]])

    angular_velocity0 = array([1,2,-3])*1e-1
    eta0 = 1
    eps0 = array([0,0,0])

    eulers = array([0,0,0])

    state0 = hstack([eulers, angular_velocity0])

    solver = ode(propagate)
    solver.set_integrator('lsoda')
    solver.set_initial_value(state0, 0)
    solver.set_f_params(INERTIA)

    newstate = []
    time = []

    tspan = 3*60
    timestep = .1

    while solver.successful() and solver.t < tspan:

        newstate.append(solver.y)
        time.append(solver.t)

        solver.integrate(solver.t + timestep)

    newstate = vstack(newstate)
    time = hstack(time)


    #Generate lightcurve

    lightcurve = []

    for state in newstate:
        eulers = state[:3]

        dcm_body2eci = R.from_euler(ROTATION, eulers).as_dcm().T

        power = 0
        for facet, area in zip(FACETS, AREAS):
            normal = dcm_body2eci@facet
            power += facet_brightness(OBS_VEC, SUN_VEC, ALBEDO, normal, area)

        lightcurve.append(power)

    lightcurve = hstack(lightcurve)



    lightcurve += random.normal(0, .0001, size = lightcurve.shape)


    #Save Data

    save('lightcurve', lightcurve)
    save('time', time)

    plt.plot(time, lightcurve)
    plt.show()


