from numpy import *
from numpy.linalg import *

from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import datetime
import sys

sys.path.insert(0, '../../Aero_Funcs')

import Aero_Funcs as AF
import Aero_Plots as AP
import Controls_Funcs as CF

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio

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
    C_SUN = 455 #Visible spectrum solar flux
    r_specular = .9
    r_diffuse = 0
    n_phong = 10
    CCD_GAIN = 4.8 #Electrons per CCD "Count"
    TELESCOPE_DIAMETER = 1 #meter
    ELECTRON_ENERGY = 2.27 #Electron Volt [eV]
    EXPOSURE_TIME = .028 #s
    J2eV = 1.6022e-19 #Joules per eV

    obs_norm = obs_vec/norm(obs_vec)
    sun_norm = sun_vec/norm(sun_vec)
    h_vec = (obs_norm + sun_norm)
    h_vec = h_vec/norm(h_vec)

    dot_ns = dot(normal, sun_norm)
    dot_no = dot(normal, obs_norm)
    dot_nh = dot(normal, h_vec)

    F_reflect = r_specular + (1-r_specular)*(1 - dot(sun_norm, h_vec))
    exponent = n_phong
    denominator = dot_ns + dot_no - dot_ns*dot_no

    specular = (n_phong+1)/(8*pi)*(dot_nh**exponent)/denominator*F_reflect

    diffuse = 28*r_diffuse/(23*pi)*(1 - r_specular)*(1 - (1 - dot_ns/2)**5)*(1 - (1 - dot_no/2)**5)

    Fsun = C_SUN*(specular + diffuse)*dot_ns
    Fobs = Fsun*area*dot_no/norm(obs_vec)**2

    # collecting_area = pi*TELESCOPE_DIAMETER**2/4
    # collected_energy = Fobs*collecting_area*EXPOSURE_TIME
    # photons_collected = collected_energy/J2eV/ELECTRON_ENERGY
    # counts = photons_collected/CCD_GAIN #flux
    # instrument_magnitude =  2.5*log10(counts/EXPOSURE_TIME)

    return Fobs


class Facet():

    def __init__(self, x_dim, y_dim, center_pos, name = '', facet2body = None, double_sided = False):
        self.center = center_pos
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.area = x_dim*y_dim
        self.double_sided = double_sided

        self.name = name

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

    def __init__(self, facets, sample_dim = .01):

        self.facets = facets
        self.obscuring_facets = {}
        self.sample_points = {}
        self.sample_nums = {}
        self.sample_dim = sample_dim

        self.calc_obscuring_faces()
        self.calc_sample_pts()

        

    def A_obscures_B(self, A, B):

        for vertex in A.vertices:
            v_test = vertex - B.center
            if dot(v_test, B.unit_normal) > 0.03:
                #print(dot(v_test, B.unit_normal), A.name, B.name)
                return True

    def calc_obscuring_faces(self):
        for B in self.facets:
            self.obscuring_facets[B] = []
            for A in self.facets:
                if (A != B) and self.A_obscures_B(A, B):
                    self.obscuring_facets[B].append(A)

    def calc_sample_pts(self):

        for facet in self.facets:

            self.sample_points[facet] = []
            x_num = ceil(facet.x_dim/self.sample_dim)
            y_num = ceil(facet.y_dim/self.sample_dim)

            self.sample_nums[facet] = x_num*y_num

            x_buff = facet.x_dim/x_num/2.0
            y_buff = facet.y_dim/y_num/2.0

            for x in linspace(-facet.x_dim/2.0 + x_buff, facet.x_dim/2.0 - x_buff, x_num):
                for y in linspace(-facet.y_dim/2.0 + y_buff, facet.y_dim/2.0 - y_buff, y_num):
                    self.sample_points[facet].append(facet.facet2body.T@array([x,y,0]))



    def calc_reflected_power(self, obs_vec_body, sun_vec_body):

        power = 0
        #poop = []
        for facet in self.facets:

            if dot(facet.unit_normal, sun_vec_body) > 0 and dot(facet.unit_normal, obs_vec_body) > 0:

                if len(self.obscuring_facets[facet]) == 0:
                    reflecting_area = facet.area
                else:
                    reflecting_area = self.calc_reflecting_area(obs_vec_body, sun_vec_body, facet)

                power += phong_brdf(obs_vec_body, sun_vec_body, facet.unit_normal, reflecting_area)
                #poop.append(facet.name)

        #print(poop)


        return power

    def calc_reflecting_area(self, obs_vec_body, sun_vec_body, facet):

        num_invisible = 0
        for pt in self.sample_points[facet]:
            if len(self.obscuring_facets[facet]) == 0:
                area = facet.area
            else:
                for obscurer in self.obscuring_facets[facet]:
                    if (obscurer.intersects(pt, obs_vec_body) != inf) or (obscurer.intersects(pt, sun_vec_body) != inf):
                        num_invisible += 1
                        break
                area = facet.area*(1 - num_invisible/self.sample_nums[facet])

        return area

    def trace_ray(self, source, ray, sun_vec):

        obs_vec = -ray
        distances = asarray([f.intersects(source, ray) for f in self.facets])
        index = where(distances == amin(distances))[0][0]

        distance = distances[index]
        facet = self.facets[index]



        if facet.double_sided and dot(facet.unit_normal, obs_vec) < 0:
            unit_normal = -facet.unit_normal
        else:
            unit_normal = facet.unit_normal
            
        # if facet.name == '+X wing':
        #     print(dot(unit_normal, sun_vec) < 0, dot(unit_normal, obs_vec) < 0)

        # if distance != inf:
        #     print(facet.name, dot(unit_normal, sun_vec) < 0, dot(unit_normal, obs_vec) < 0)
        #     print(facet.name, facet.unit_normal)


        if distance == inf:
            return 0

        elif dot(unit_normal, sun_vec) < 0 or dot(unit_normal, obs_vec) < 0:
            return 0

        else:

            surface_pt = source + ray*distance
            if len(self.obscuring_facets[facet]) != 0:
                for obscurer in self.obscuring_facets[facet]:
                        dist = obscurer.intersects(surface_pt, sun_vec)
                        if (dist != inf) and (dist > 0):
                            #print(facet.name)
                            return 0
                            break
            return phong_brdf(obs_vec, sun_vec, unit_normal, 1)

        

def generate_image(spacecraft_geometry, obs_vec_body, sun_vec_body, win_dim = (2,2), dpm = 50):
    """
    camera_axis is the direction that the camera is pointing
    sun axis is the direction that the light is moving (sun to spacecraft)
    """

    win_pix = (win_dim[0]*dpm, win_dim[1]*dpm)
    image = zeros(win_pix)
    perspective_distance = 5

    obs_vec_body = obs_vec_body/norm(obs_vec_body)
    camera_pos = obs_vec_body*5
    ray_direction = -obs_vec_body

    
    camera_angle = arccos(dot(ray_direction, array([0,0,1])))
    camera_rotation = CF.axis_angle2dcm(cross(array([0,0,1]), ray_direction), camera_angle)

    for y, row in enumerate(image):
        for x, pix in enumerate(row):
            x_pos = (x - win_pix[0]/2)/dpm
            y_pos = (win_pix[1]/2 - y)/dpm
            pix_pos = camera_rotation@array([x_pos, y_pos, 0]) + camera_pos
            
            image[x,y] = spacecraft_geometry.trace_ray(pix_pos, ray_direction, sun_vec_body)

    m = amax(image)
    # if m == 0:
    #     print('m = 0',obs_vec_body, sun_vec_body)
    if m != 0:
        image = image/m

    return image


class Premade_Spacecraft():

    def __init__(self):
        pZ = Facet(1, 1, array([0,0, .5]), facet2body = identity(3) , name = '+Z')
        nZ = Facet(1, 1, array([0,0,-.5]), facet2body = CF.Cy(pi) , name = '-Z')
        pX = Facet(1, 1, array([ .5,0,0]), facet2body = CF.Cy(pi/2), name = '+X')
        nX = Facet(1, 1, array([-.5,0,0]), facet2body = CF.Cy(-pi/2), name = '-X')
        pY = Facet(1, 1, array([0, .5,0]), facet2body = CF.Cx(-pi/2), name = '+Y')
        nY = Facet(1, 1, array([0,-.5,0]), facet2body = CF.Cx(pi/2), name = '-Y')
        wingnX = Facet(1, .5, array([-1, 0,0]), facet2body = CF.Cx(pi/2), name = '-X wing', double_sided = True)
        wingpX = Facet(1, .5, array([ 1, 0,0]), facet2body = CF.Cx(pi/2), name = '+X wing', double_sided = True)
        
        self.BOX_WING = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ,wingnX, wingpX], sample_dim = .1)

        self.BOX = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .1)

        plate = Facet(1,1, array([0,0,0]), facet2body = identity(3), name = 'plate', double_sided = True)
        self.PLATE = Spacecraft_Geometry([plate])


        segments = 20
        angle = 2*pi/segments
        radius = 1.5
        side_length = 1.8*sin(angle/2)*radius
        lenght = 9
        cylinder_facets = []
        for theta in linspace(0, 2*pi, segments)[:-1]:
            pos = CF.Cz(theta)@array([1,0,0])
            facet2body = CF.Cz(theta)@CF.Cy(pi/2)
            cylinder_facets.append(Facet(lenght, side_length, pos, facet2body = facet2body, name = 'cylinder'))

        pZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,lenght/2]), name = '+Z', facet2body = identity(3))
        nZ = Facet(1.4/sqrt(2), 1.4/sqrt(2), array([0,0,-lenght/2]), name = '-Z', facet2body = CF.Cx(pi))
        cylinder_facets.append(pZ)
        cylinder_facets.append(nZ)

        self.CYLINDER = Spacecraft_Geometry(cylinder_facets, sample_dim = .5)

        pZ = Facet(3.0, 1.0, array([0,0, 1.0]), facet2body = identity(3) , name = '+Z')
        nZ = Facet(3.0, 1.0, array([0,0,-1.0]), facet2body = CF.Cy(pi) , name = '-Z')
        pX = Facet(2.0, 1.0, array([ 1.5,0,0]), facet2body = CF.Cy(pi/2), name = '+X')
        nX = Facet(2.0, 1.0, array([-1.5,0,0]), facet2body = CF.Cy(-pi/2), name = '-X')
        pY = Facet(3.0, 2.0, array([0, .5,0]), facet2body = CF.Cx(-pi/2), name = '+Y')
        nY = Facet(3.0, 2.0, array([0,-.5,0]), facet2body = CF.Cx(pi/2), name = '-Y')

        self.RECTANGLE = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .1)

        pZ = Facet(5.0, 1.0, array([0,0, 1.0]), facet2body = identity(3) , name = '+Z')
        nZ = Facet(5.0, 1.0, array([0,0,-1.0]), facet2body = CF.Cy(pi) , name = '-Z')
        pX = Facet(2.0, 1.0, array([ 2.5,0,0]), facet2body = CF.Cy(pi/2), name = '+X')
        nX = Facet(2.0, 1.0, array([-2.5,0,0]), facet2body = CF.Cy(-pi/2), name = '-X')
        pY = Facet(5.0, 2.0, array([0, .5,0]), facet2body = CF.Cx(-pi/2), name = '+Y')
        nY = Facet(5.0, 2.0, array([0,-.5,0]), facet2body = CF.Cx(pi/2), name = '-Y')

        self.LONG_RECTANGLE = Spacecraft_Geometry([pX,nX,pY,nY,pZ,nZ], sample_dim = .1)

    def get_geometry(self, name):
        name = name.upper()
        if name == 'BOX_WING':
            return self.BOX_WING
        elif name == "BOX":
            return self.BOX
        elif name == 'PLATE':
            return self.PLATE
        elif name == 'CYLINDER':
            return self.CYLINDER
        elif name == 'RECTANGLE':
            return self.RECTANGLE
        elif name == 'LONG_RECTANGLE':
            return self.LONG_RECTANGLE
        else:
            print(name, 'is not a valid geometry.')


def loading_bar(decimal_percentage, text = ''):
    bar = '#'*int(decimal_percentage*20)
    print('{2} Loading:[{0:<20}] {1:.1f}%'.format(bar,decimal_percentage*100, text), end = '\r')
    if decimal_percentage == 1:
        print('')

if __name__ == '__main__':

    obs_vecs = load('Splotchy_Rectangle_Leo/obs_vecs.npy')[::50]
    sun_vecs = load('Splotchy_Rectangle_Leo/sun_vecs.npy')[::50]
    attitudes = load('Splotchy_Rectangle_Leo/true_attitude.npy')[::50]

    SC = Premade_Spacecraft().SPLOTCHY_RECTANGLE

    # screwy_attitude = load('LEO_RECTANGLE/estimated_states.npy')[34834]
    # screwy_obs_vec = obs_vecs[34834]
    # dcm_eci2body = CF.euler2dcm(ROTATION, screwy_attitude[0:3]).T
    # image = generate_image(SC, dcm_eci2body@screwy_obs_vec, dcm_eci2body@sun_vec_body, win_dim = (4,4), dpm = 5)
    # power = SC.calc_reflected_power(dcm_eci2body@screwy_obs_vec, dcm_eci2body@sun_vec_body)
    # print(power)
    # plt.imshow(image, cmap= 'Greys')
    # plt.show()


    for facet in SC.obscuring_facets:
        print(facet.name, [f.name for f in SC.obscuring_facets[facet]])
        print(facet.name, facet.center, facet.unit_normal)

    num_frames = len(obs_vecs)
    print(num_frames)

    # obs_body = array([ 0.67027951, -0.02873575, -0.74155218])
    # sun_body = array([-0.01285413, -0.99098831,  0.13332702])
    # image = generate_image(SC, obs_body, sun_body, win_dim = (4,4), dpm = 5)
    # plt.imshow(image, cmap = 'Greys')
    # plt.show()

    frames = []
    count = 0
    max_val = 0
    for mrps, obs_vec, sun_vec in zip(attitudes[:,0:3], obs_vecs, sun_vecs):
        dcm_eci2body = CF.euler2dcm(ROTATION, mrps).T
        #dcm_eci2body = CF.mrp2dcm(mrps).T
        image = generate_image(SC, dcm_eci2body@obs_vec, dcm_eci2body@sun_vec, win_dim = (4,4), dpm = 5)
        im_max = amax(image)
        if im_max > max_val:
            max_val = im_max
        frames.append(image)
        count += 1
        loading_bar(count/num_frames, 'Rendering gif')

    frames = [frame/max_val for frame in frames]

    imageio.mimsave('./true_rotation.gif',frames, fps = 10)

    #plt.imshow(image, cmap = 'gray')
    # plt.show()

                









