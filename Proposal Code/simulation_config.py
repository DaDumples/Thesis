from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R

import read_lightcurve_data as rlc

class VARIABLES():
    ROTATION = 'xyz'
    DT = .15

    __passes = rlc.SpacecraftObservation('CalPolySLO_obs.csv')
    __passes.find_tle('sat41788.txt')
    PASS = __passes.Passes[0]
    LATITUDE = 37.1348 #degrees North
    LONGITUDE = -12.2110 #degrees East
    ALTITUDE = 684 #meters

class TRUTH():
    X_LEN = 1 #Zaxis
    Y_LEN = 4 #Yaxis
    Z_LEN = 10 #Xaxis
    MASS = 50 #k
    Ixx = (1/12)*MASS*(Y_LEN**2 + Z_LEN**2)
    Iyy = (1/12)*MASS*(X_LEN**2 + Z_LEN**2)
    Izz = (1/12)*MASS*(X_LEN**2 + Y_LEN**2)
    
    DCM_OFFSET = R.from_euler('xyz', array([0.01, 0.01, 0.01])).as_dcm()

    INERTIA = DCM_OFFSET@diag(array([Ixx, Iyy, Izz]))@DCM_OFFSET.T

    FACETS = [array([ 1, 0, 0]),
              array([-1, 0, 0]),
              array([ 0, 1, 0]),
              array([ 0,-1, 0]),
              array([ 0, 0, 1]),
              array([ 0, 0,-1])]

    MEASUREMENT_VARIANCE = .001
    ALBEDO = .6
    AREAS = array([Y_LEN*Z_LEN,Y_LEN*Z_LEN,
                   X_LEN*Z_LEN,X_LEN*Z_LEN,
                   X_LEN*Y_LEN,X_LEN*Y_LEN])

    OBS_VEC = array([1,1,0])/norm(array([1,1,0]))
    SUN_VEC = array([1,0,0])




class ESTIMATE():
    FACETS = [array([ 1, 0, 0]),
              array([-1, 0, 0]),
              array([ 0, 1, 0]),
              array([ 0,-1, 0]),
              array([ 0, 0, 1]),
              array([ 0, 0,-1])]

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
    AREAS = TRUTH.AREAS
    OBS_VEC = TRUTH.OBS_VEC
    SUN_VEC = TRUTH.SUN_VEC
