import numpy as np
np.seterr(all='raise')

class ConstantVelocityModel():
    """ State space: x, y, vx, vy

    Methods:
        self.F: Returns the motion Jacobian matrix
        self.f: Returns state prediction
        self.Q: Returns discrete time motion noise
    """

    dim = 4
    def __init__(self, T, sigma_velocity):
        self.T = T
        self.F = lambda x: np.array([[1, 0, T, 0],
                                    [0, 1, 0, T],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        self.Q = sigma_velocity**2 *  np.array([[T**4/4, 0, T**3/2, 0],
                                    [0, T**4/4, 0, T**3/2],
                                    [T**3/2, 0, T**2, 0],
                                    [0, T**3/2, 0, T**2]])
        self.f = lambda x: self.F(x) @ x


class CoordinatedTurnModel():
    """ State space: x, y, v, heading, turn-rate.
    
    Methods:
        self.F: Returns the motion Jacobian matrix
        self.f: Returns state prediction
        self.Q: Returns discrete time motion noise
    """

    def __init__(self, T, sigma_velocity, sigma_turnrate):
        self.T = T

        Q = np.zeros((5,5))
        Q[2,2] = T*sigma_velocity**2
        Q[4,4] = T*sigma_turnrate**2
        self.Q = Q

    def f(self, x):
        x = np.reshape(x, (5,1))

        f = x + np.array([
            [(self.T * x[2] * np.cos(x[3]))[0]],
            [(self.T * x[2] * np.sin(x[3]))[0]],
            [0],
            [(self.T * x[4])[0]],
            [0],
        ])
        return f

    def F(self, x):
        x = np.reshape(x, (5,1))
        F = np.array([
            [1, 0, (self.T*np.cos(x[3]))[0], (-self.T*x[2]*np.sin(x[3]))[0], 0],
            [0, 1, (self.T*np.sin(x[3]))[0], (self.T*x[2]*np.cos(x[3]))[0], 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, self.T],
            [0, 0, 0, 0, 1],
        ])
        return F