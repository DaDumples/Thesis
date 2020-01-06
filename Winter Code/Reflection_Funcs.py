from numpy import *
from numpy.linalg import *

from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import datetime
import sys

from shapely.geometry import Polygon
from shapely.ops import cascaded_union

sys.path.insert(0, '../../Aero_Funcs')

import read_lightcurve_data as rlc
import simulation_config as config
import Aero_Funcs as AF
import Aero_Plots as AP
import Controls_Funcs as CF

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
R_SPECULAR = VARIABLES.R_SPECULAR
R_DIFFUSION = VARIABLES.R_DIFFUSION
N_PHONG = VARIABLES.N_PHONG
C_SUN = VARIABLES.C_SUN

def lommel_seeliger(obs_vec, sun_vec, albedo, normal, area):
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


def phong_brdf(obs_vec, sun_vec, normal, area):
    #As implemented in INACTIVE SPACE OBJECT SHAPE ESTIMATION
    #VIA ASTROMETRIC AND PHOTOMETRIC DATA FUSION

    #Assumes specular lobe is even in all directions, nu = nv = N_PHONG

    obs_norm = obs_vec/norm(obs_vec)
    sun_norm = sun_vec/norm(sun_vec)
    h_vec = (obs_norm + sun_norm)
    h_vec = h_vec/norm(h_vec)

    dot_ns = dot(normal, sun_norm)
    dot_no = dot(normal, obs_norm)
    dot_nh = dot(normal, h_vec)

    F_reflect = R_SPECULAR + (1-R_SPECULAR)*(1 - dot(sun_norm, h_vec))
    exponent = N_PHONG
    denominator = dot_ns + dot_no - dot_ns*dot_no

    specular = (N_PHONG+1)/(8*pi)*(dot_nh**exponent)/denominator*F_reflect

    diffuse = 28*R_DIFFUSION/(23*pi)*(1 - R_SPECULAR)*(1 - (1 - dot_ns/2)**5)*(1 - (1 - dot_no/2)**5)

    Fsun = C_SUN*(specular + diffuse)*dot_ns
    Fobs = Fsun*area*dot_no/norm(obs_vec)**2

    return Fobs


def states_to_lightcurve(times, states, pass_obj, quats = False):

    lightcurve = []
    spacecraft = twoline2rv(pass_obj['TLE'][0], pass_obj['TLE'][1], wgs84)
    telescope_ecef = AF.lla_to_ecef(LAT, LON, ALT, geodetic = True)
    utc0 = pass_obj['EPOCH']

    sc_posses = []

    for time, state in zip(times, states):

        now = utc0 + datetime.timedelta(seconds = time)

        sun_vec = AF.vect_earth_to_sun(now)


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
            dcm_body2eci = CF.quat2dcm(eps, eta)
        else:
            eulers = state[:3]
            dcm_body2eci = CF.euler2dcm(ROTATION, eulers)

        power = 0
        for facet, area in zip(FACETS, AREAS):
            normal = dcm_body2eci@facet

            if dot(normal, sun_vec) > 0 and dot(normal, obs_vec) > 0:
                power += phong_brdf(obs_vec, sun_vec, normal, area)

        lightcurve.append(power)
        sc_posses.append(sc_eci)

    lightcurve = hstack(lightcurve)
    sc_posses = vstack(sc_posses)

    return lightcurve, sc_posses


class Facet():

    def __init__(self, x_dim, y_dim, center_pos, name = '', albedo = .6, facet2body = None):
        self.center = center_pos
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.area = x_dim*y_dim

        self.name = name
        self.albedo = albedo

        # angle = arccos(dot(self.unit_normal, array([0,0,1])))
        # axis = cross(self.unit_normal, array([0,0,1]))
        # if facet2body == None:
        #     self.facet2body = axis_angle2dcm(-axis, angle)
        #     self.unit_normal = facet2body@array([0,0,1])
        # else:
        self.facet2body = facet2body
        self.unit_normal = facet2body@array([0,0,1])
        self.vertices = []

        self.calc_vertices()


    def intersects(self, source, direction, check_bounds = True):


        direction = direction/norm(direction)

        if dot(direction, self.unit_normal) == 0:
            return inf
        elif dot((self.center - source), self.unit_normal) == 0:
            return 0

        else:
            distance_from_source = dot((self.center - source), self.unit_normal)/dot(direction, self.unit_normal)



            intersection = source + direction*distance_from_source

            intersection_in_plane = self.facet2body.T@(intersection - self.center)

            if check_bounds:
                within_x = (intersection_in_plane[0] <= self.x_dim/2) and (intersection_in_plane[0] >= -self.x_dim/2)
                within_y = (intersection_in_plane[1] <= self.y_dim/2) and (intersection_in_plane[1] >= -self.y_dim/2)

                if within_x and within_y:
                    return distance_from_source
                else:
                    return inf
            else:
                return distance_from_source

    def calc_vertices(self):
        for x in [-self.x_dim/2, self.x_dim/2]:
            for y in [-self.y_dim/2, self.y_dim/2]:
                pos = self.facet2body@array([x, y, 0]) + self.center
                self.vertices.append(pos)
        self.vertices = vstack(self.vertices)


class Spacecraft_Geometry():

    def __init__(self, faces, sample_dim = .01):

        self.faces = faces
        self.obscuring_faces = {}
        self.sample_points = {}
        self.sample_nums = {}
        self.sample_dim = sample_dim

        self.calc_obscuring_faces()
        self.calc_sample_pts()

        

    def A_obscures_B(self, A, B):

        for vertex in A.vertices:
            v_test = vertex - B.center
            if dot(v_test, B.unit_normal) > 0:
                return True

    def calc_obscuring_faces(self):
        for B in self.faces:
            self.obscuring_faces[B] = []
            for A in self.faces:
                if (A != B) and self.A_obscures_B(A, B):
                    self.obscuring_faces[B].append(A)

    def calc_sample_pts(self):

        for face in self.faces:

            self.sample_points[face] = []
            x_num = int(face.x_dim/self.sample_dim)
            y_num = int(face.y_dim/self.sample_dim)

            self.sample_nums[face] = x_num*y_num

            x_buff = face.x_dim/x_num/2
            y_buff = face.y_dim/y_num/2

            for x in linspace(-face.x_dim/2 + x_buff, face.x_dim/2 - x_buff, x_num):
                for y in linspace(-face.y_dim/2 + y_buff, face.y_dim/2 - y_buff, y_num):
                    self.sample_points[face].append(face.facet2body.T@array([x,y,0]))



    # def calc_reflected_power(self, obs_vec_body, sun_vec_body):

    #     power = 0
    #     for facet in zip(self.faces):

    #         if dot(normal, sun_vec) > 0 and dot(normal, obs_vec) > 0:

    #             if len(self.obscuring_faces[facet]) == 0:
    #                 power += phong_brdf(obs_vec, sun_vec, normal, area - blocked_area)

    #             else:

    #                 polygons = []
    #                 for obscuring_face in self.obscuring_faces[facet]:
    #                     behind_vertices = []
    #                     shadow_vertices = []
    #                     for obscuring_vertex in obscuring_face.vertices:
    #                         behind_vertices.append(facet.intersects(obscuring_vertex, obs_vec_body, check_bounds = False)*obs_vec_body + source)
    #                         shadow_vertices.append(facet.intersects(obscuring_vertex, sun_vec_body, check_bounds = False)*sun_vec_body + source)
    #                     polygons.append(Polygon(behind_vertices).convex_hull)
    #                     polygons.append(Polygon(shadow_vertices).convex_hull)

    #                 blocked_area = facet.intersection(cascaded_union(polygons)).area

    #                 power += phong_brdf(obs_vec, sun_vec, normal, area - blocked_area)

    def calc_reflecting_area(self, obs_vec_body, sun_vec_body, facet):

        num_visible = 0
        for pt in self.sample_points[facet]:
            for obscurer in self.obscuring_faces[facet]:
                if (obscurer.intersects(pt, obs_vec_body) == inf) and (obscurer.intersects(pt, sun_vec_body) == inf):
                    num_visible += 1
                    break

        return facet.area*num_visible/self.sample_nums[facet]


if __name__ == '__main__':

    face1 = Facet(1, 1, array([0,0,0]), facet2body = identity(3))
    face2 = Facet(1, 1, array([-.5,0,0]), facet2body = CF.Cy(-pi/2))

    tbone = Spacecraft_Geometry([face1, face2])

    obs_vec = array([-1,1,1])/norm(array([-1,1,1]))
    sun_vec = array([-1,-1,1])/norm(array([-1,-1,1]))

    print(face2.intersects(array([.5,0,0]), sun_vec))
    print(tbone.calc_reflecting_area(obs_vec, sun_vec, tbone.faces[0]))

                









