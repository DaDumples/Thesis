import PIL
from numpy import *
from numpy.linalg import *
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

R_DIFFUSION = .5 #metal has zero diffusion
R_SPECULAR = .05 #gueess
N_PHONG = 10 #guess for now, see PHONG BRDF
C_SUN = 255 #w/m^2, power of visible spectrum

class Facet():

    def __init__(self, x_dim, y_dim, unit_normal, center_pos):
        self.center = center_pos
        self.unit_normal = unit_normal/norm(unit_normal)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.area = x_dim*y_dim

        angle = arccos(dot(self.unit_normal, array([0,0,1])))
        axis = cross(array([0,0,1]), self.unit_normal)
        self.rotation = R.from_rotvec(angle*axis)


    def intersects(self, source, direction):


        direction = direction/norm(direction)

        if dot(direction, self.unit_normal) == 0:
            return inf
        elif dot((self.center - source), self.unit_normal) == 0:
            return 0

        else:
            distance_from_source = dot((self.center - source), self.unit_normal)/dot(direction, self.unit_normal)



            intersection = source + direction*distance_from_source

            intersection_in_plane = self.rotation.as_dcm().T@(intersection - self.center)

            within_x = (intersection_in_plane[0] <= self.x_dim) and (intersection_in_plane[0] >= -self.x_dim)
            within_y = (intersection_in_plane[1] <= self.y_dim) and (intersection_in_plane[1] >= -self.y_dim)

            if within_x and within_y:
                return distance_from_source
            else:
                return inf

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


face1 = Facet(1, 1, array([1,0,0]), array([1,0,0]))
face2 = Facet(1, 1, array([-1,0,0]), array([-1,0,0]))
face3 = Facet(1, 1, array([0,1,0]), array([0,1,0]))
face4 = Facet(1, 1, array([0,-1,0]), array([0,-1,0]))
face5 = Facet(1, 1, array([0,0,1]), array([0,0,1]))
face6 = Facet(1, 1, array([0,0,-1]), array([0,0,-1]))

facets = [face1, face2, face3, face4, face5, face6]

win_dim = (4, 4) #(meters, meters)
dpm = 150 #dots per meter
win_pix = (win_dim[0]*dpm, win_dim[1]*dpm)
image = zeros(win_pix)

camera_pos = array([3.1, 3, 3.2])
camera_axis = -camera_pos/norm(camera_pos)
camera_angle = arccos(dot(camera_axis, array([0,0,1])))
camera_rotation = R.from_rotvec(camera_angle*cross(array([0,0,1]), camera_axis))
sun_pos = array([5, 5, 4])
perspective_distance = 5


# for f in facets:
#     print(f.intersects(array([4,4,4]), array([-4,-4,-4])/norm(array([-4,-4,-4]))))


for y, row in enumerate(image):
    for x, pix in enumerate(row):
        x_pos = (x - win_pix[0]/2)/dpm
        y_pos = (win_pix[1]/2 - y)/dpm
        pix_pos = camera_rotation.as_dcm()@array([x_pos, y_pos, 0]) + camera_pos
        distances = asarray([f.intersects(pix_pos, camera_axis) for f in facets])
        index = where(distances == amin(distances))[0][0]

        ray = pix_pos - (camera_pos/norm(camera_pos)*perspective_distance + camera_pos)
        ray = ray/norm(ray)

        distance = facets[index].intersects(pix_pos, ray)

        if distance == inf:
            image[x, y] = 0
        else:
            obs_vec = camera_pos + pix_pos - facets[index].center
            sun_vec = sun_pos - facets[index].center
            image[x, y] = phong_brdf(obs_vec, sun_vec, facets[index].unit_normal, facets[index].area)
            #print(image[x,y])

m = amax(image)
image = image/m*255

plt.imshow(-image, cmap = 'Greys')
plt.show()



