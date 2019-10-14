import matplotlib.pyplot as plt
from numpy import *

truth = load('true_states.npy')
time = load('time.npy')
est = load('estimated_states.npy')


error = linalg.norm(truth[:,-3:] - est[:,-3:], axis = 1)
plt.plot(error)
plt.show()