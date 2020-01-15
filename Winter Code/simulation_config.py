from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R

import read_lightcurve_data as rlc

class VARIABLES():
    ROTATION = 'xyz'
    DT = .01

    __passes = rlc.SpacecraftObservation('CalPolySLO_obs.csv')
    __passes.find_tle('sat41788.txt')
    PASS = __passes.Passes[0]
    LATITUDE = 37.1348 #degrees North
    LONGITUDE = -12.2110 #degrees East
    ALTITUDE = 684 #meters

    R_DIFFUSION = .5 #metal has zero diffusion
    R_SPECULAR = .05 #gueess
    N_PHONG = 10 #guess for now, see PHONG BRDF
    C_SUN = 455 #w/m^2, power of visible spectrum

    

class TRUTH():
    X_LEN = 1 #Zaxis
    Y_LEN = 3 #Yaxis
    Z_LEN = 2 #Xaxis
    MASS = 50 #k
    Ixx = (1/12)*MASS*(Y_LEN**2 + Z_LEN**2)
    Iyy = (1/12)*MASS*(X_LEN**2 + Z_LEN**2)
    Izz = (1/12)*MASS*(X_LEN**2 + Y_LEN**2)
    
    DCM_OFFSET = R.from_euler('xyz', array([0.01, 0.01, 0.01])).as_dcm()

    #INERTIA = DCM_OFFSET@diag(array([Ixx, Iyy, Izz]))@DCM_OFFSET.T
    INERTIA = identity(3)*MASS/12

    # FACETS = [array([ 1, 0, 0]),
    #           array([-1, 0, 0]),
    #           array([ 0, 1, 0]),
    #           array([ 0,-1, 0]),
    #           array([ 0, 0, 1]),
    #           array([ 0, 0,-1])]

    MEASUREMENT_VARIANCE = .001
    ALBEDO = .6



class ESTIMATE():
    X_LEN = TRUTH.X_LEN #Zaxis
    Y_LEN = TRUTH.X_LEN #Yaxis
    Z_LEN = TRUTH.X_LEN #Xaxis
    MASS = 48 #k
    Ixx = (1/12)*MASS*(Y_LEN**2 + Z_LEN**2)
    Iyy = (1/12)*MASS*(X_LEN**2 + Z_LEN**2)
    Izz = (1/12)*MASS*(X_LEN**2 + Y_LEN**2)

    INERTIA = diag(array([Ixx, Iyy, Izz]))

    MEASUREMENT_VARIANCE = TRUTH.MEASUREMENT_VARIANCE
    ALBEDO = TRUTH.ALBEDO

def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1], A[0], 0]])

def propagate_eulers(t, state, inertia):
    phi = state[0]
    theta = state[1]
    #zeta = state[2]
    omega = state[3:6]

    A = array([[1, sin(phi)*tan(theta)  , cos(phi)*tan(theta) ],
               [0, cos(phi)             , -sin(phi)           ],
               [0, sin(phi)*(1/cos(theta)) , cos(phi)*(1/cos(theta)) ]])

    deulers = A@omega

    domega = -inv(inertia)@crux(omega)@inertia@omega

    return hstack([deulers, domega])

def propagate_quats(t, state, inertia):
    eta = state[0]
    eps = state[1:4]
    omega = state[4:]

    deps = .5*(eta*identity(3) + crux(eps))@omega
    deta = -.5*dot(eps, omega)
    domega = -inv(inertia)@crux(omega)@inertia@omega

    derivatives = hstack([deta, deps, domega])

    return derivatives