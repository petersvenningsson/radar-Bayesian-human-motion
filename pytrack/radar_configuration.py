import numpy as np

_angles = [-k*np.pi/4 for k in range(5)]
_radius = 4.38/2 + 1
pulseON_positions = [np.array([_radius*np.cos(angle),_radius*np.sin(angle)]) for angle in _angles]
