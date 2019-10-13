import matplotlib.pyplot as plt
from numpy import *

truth = load('true_states.npy')
time = load('time.npy')
est = load('estimated_states.npy')

plt.plot(truth[:,-3:])
plt.plot(est[:,-3:])
plt.show()