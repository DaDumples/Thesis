import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import *
from numpy.linalg import *

facet = array([[0,0], [0,1], [1,1], [1,0], [0,0]])

# fig = plt.figure()
# ax = Axes3D(fig)

# ax.plot(facet[:,0] - .5, facet[:,1] - .5, zeros(5), 'k')
# ax.quiver(0,0,0, 1, 0, 0, color ='k')
# ax.text(1,0,0, r'$\mathrm{u}_x$', fontsize = 12)
# ax.quiver(0,0,0, 0, 1, 0, color = 'k')
# ax.text(0,1,0, r'$\mathrm{u}_y$', fontsize = 12)
# ax.quiver(0,0,0, 0, 0, 1, color ='k')
# ax.text(0,0,1, r'$\mathrm{u}_n$', fontsize = 12)

# sun = array([-.7, .5, .5])
# sun = sun/norm(sun)

# ax.quiver(0,0,0, *sun, color ='r')
# ax.text(*sun, r'$\mathrm{u}_{sun}$', fontsize = 12)

# obs = array([-1, -1, 1])
# obs = obs/norm(obs)

# ax.quiver(0,0,0, *obs, color ='r')
# ax.text(*obs, r'$\mathrm{u}_{obs}$', fontsize = 12)


# h = (sun + obs)/2
# ax.quiver(0,0,0, *h, color ='g')
# ax.text(*h, r'$\mathrm{u}_{h}$', fontsize = 12)

# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# ax.set_zlim(-1,1)
# ax.axis("off")

# plt.savefig('facet.png', dpi = 300, bbox_inches = 'tight')



# fig = plt.figure()
# ax = Axes3D(fig, azim = -110, elev = 30)
# ax.plot(facet[:,0] - .5, facet[:,1] - .5, zeros(5), 'k')
# ax.plot(.5*ones(5), facet[:,0] - .5, facet[:,1] - .5, 'k')

# ax.quiver(0,0,0, 0, 0, 1, color ='k')
# ax.text(0,0,1, r'$\mathrm{u}_n$', fontsize = 12)

# ax.quiver(0,0,0, .5, .5, .5, color ='r')
# ax.text(.5,.5,.5, r'$V_p - B_c$', fontsize = 12)

# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# ax.set_zlim(-1,1)
# ax.axis("off")

# plt.savefig('occlusion_check.png', dpi = 300, bbox_inches = 'tight')

# fig = plt.figure()
# ax = Axes3D(fig, azim = -130, elev = 30)
# ax.plot(facet[:,0] - .5, facet[:,1] - .5, zeros(5), 'k')
# ax.plot(.5*ones(5), facet[:,0] - .5, facet[:,1] - .5, 'k')

# pts = linspace(-.5, .5, 6)[1:] - 1/6/2
# for x in pts:
#     for y in pts:
#         ax.scatter(x, y, 0, color = 'k', alpha = .5)

# ax.quiver(.5, 0, 0, 0, 1, 0, color = 'k', arrow_length_ratio=.1, alpha = .5)
# ax.text(.5,1,0, r'$\mathrm{u}_y$', fontsize = 12)
# ax.quiver(.5, 0, 0, 0, 0, 1, color = 'k', arrow_length_ratio=.1, alpha = .5)
# ax.text(.5,0,1, r'$\mathrm{u}_x$', fontsize = 12)

# pt = array([pts[2], pts[2], 0])
# px_sun = array([.5, -.3, .1])
# ax.quiver(*pt, *(px_sun - pt), color ='k', arrow_length_ratio=.1)
# ax.text(px_sun[0], -.1, px_sun[2], r'$p_{\mathrm{x}, sun}$', fontsize = 12)
# ax.scatter(*px_sun, 'o', color = 'r')

# px_obs = array([.5, -.2, .7])
# ax.quiver(*pt, *(px_obs - pt), color ='k', arrow_length_ratio=.1)
# ax.text(*px_obs*1.1, r'$p_{\mathrm{x}, obs}$', fontsize = 12)
# ax.scatter(*px_obs, 'o', color = 'g')

# line1 = vstack([px_sun, px_sun])
# line1[0,1] = 0
# line2 = vstack([px_sun, px_sun])
# line2[0,2] = 0
# ax.plot(line1[:,0], line1[:,1], line1[:,2], '--r', alpha = .5)
# ax.plot(line2[:,0], line2[:,1], line2[:,2], '--r', alpha = .5)

# line1 = vstack([px_obs, px_obs])
# line1[0,1] = 0
# line2 = vstack([px_obs, px_obs])
# line2[0,2] = 0
# ax.plot(line1[:,0], line1[:,1], line1[:,2], '--r', alpha = .5)
# ax.plot(line2[:,0], line2[:,1], line2[:,2], '--r', alpha = .5)

# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# ax.set_zlim(-1,1)
# ax.axis("off")

# plt.savefig('visible_and_illuminated.png', dpi = 300, bbox_inches = 'tight')

fig = plt.figure()
ax = Axes3D(fig, azim = -48, elev = 24)
#ax.plot(facet[:,0] - .5, facet[:,1] - .5, zeros(5), 'k')
obs = array([1,0,0])
obs = obs/norm(obs)
sun = array([1,-1,0])
sun = sun/norm(sun)

Z = cross(sun, obs)/norm(cross(sun, obs))
Y = cross(Z,obs)/norm(cross(Z, obs))

ax.quiver(0, 0, 0, *obs*.5, color = 'b', arrow_length_ratio=.1, alpha = 1)
ax.quiver(0, 0, 0, *sun*.5, color = 'b', arrow_length_ratio=.1, alpha = 1)
ax.quiver(0, 0, 0, *Z, color = 'k', arrow_length_ratio=.1, alpha = .5)
ax.quiver(0, 0, 0, *Y, color = 'k', arrow_length_ratio=.1, alpha = .5)
ax.quiver(0, 0, 0, *obs, color = 'k', arrow_length_ratio=.1, alpha = .5)
ax.text(*obs*.5*1.1, r'$u_{obs}$', fontsize = 12)
ax.text(*sun*.5*1.1, r'$u_{sun}$', fontsize = 12)
ax.text(*obs*1.1, r'$X$', fontsize = 12)
ax.text(*Y*1.1, r'$Y$', fontsize = 12)
ax.text(*Z*1.1, r'$Z$', fontsize = 12)

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
ax.axis("off")

plt.savefig('Observation_Frame.png', dpi = 300, bbox_inches = 'tight')


plt.show()