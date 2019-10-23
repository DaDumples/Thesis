from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R

class VARIABLES():
    ROTATION = 'xyz'
    DT = .1

class TRUTH():
    X_LEN = 2 #Zaxis
    Y_LEN = 3 #Yaxis
    Z_LEN = 1 #Xaxis
    MASS = 50 #k
    Ixx = (1/12)*MASS*(Y_LEN**2 + Z_LEN**2)
    Iyy = (1/12)*MASS*(X_LEN**2 + Z_LEN**2)
    Izz = (1/12)*MASS*(X_LEN**2 + Y_LEN**2)
    
    DCM_OFFSET = R.from_euler('xyz', array([.01, .01, -.01])).as_dcm()

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

    OBS_VEC = array([2,1,3])/norm(array([2,1,3]))
    SUN_VEC = array([1,0,0])




class ESTIMATE():
    FACETS = [array([ 1, 0, 0]),
              array([-1, 0, 0]),
              array([ 0, 1, 0]),
              array([ 0,-1, 0]),
              array([ 0, 0, 1]),
              array([ 0, 0,-1])]

    X_LEN = 2 #Zaxis
    Y_LEN = 3 #Yaxis
    Z_LEN = 1 #Xaxis
    MASS = 48 #k
    Ixx = (1/12)*MASS*(Y_LEN**2 + Z_LEN**2)
    Iyy = (1/12)*MASS*(X_LEN**2 + Z_LEN**2)
    Izz = (1/12)*MASS*(X_LEN**2 + Y_LEN**2)

    INERTIA = diag(array([Ixx, Iyy, Izz]))

    MEASUREMENT_VARIANCE = .001
    ALBEDO = .6
    AREAS = TRUTH.AREAS
    OBS_VEC = array([2,1,3])/norm(array([2,1,3]))
    SUN_VEC = array([1,0,0])
