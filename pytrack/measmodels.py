import numpy as np
import sympy as sym

class PositionMeasurmentModel():
    """ Sensor measures position state variables. 
    It is assumed that x[0] -> position in x
                       x[1] -> position in y
    """
    dim = 2

    def __init__(self, sensor_bias, sigma_position, state_dim):
        """ Input:
                sigma_position: Standard deviation of zero mean Gaussian measurement noise.
                state_dim: dimensionality of the state vector.
                sensor_bias: E.g position of sensor. Numpy array of shape (state_dim,)
        """
        self.state_dim = state_dim
        self.R = np.diag([sigma_position**2, sigma_position**2])

        self.sensor_bias = np.zeros(state_dim)
        self.sensor_bias = sensor_bias


    def H(self, x):
        """ Returns the measurement model Jacobian. 
        """
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0,0] = 1
        H[1,1] = 1
        return H
    

    def h(self, x):
        return np.reshape(x[0:2], (2,1)) - np.reshape(self.sensor_bias[0:2], (2,1))


class RangeDopplerMeasurmentModel():
    """ Sensor measures position state variables. 
    It is assumed that x[0] -> position in x
                       x[1] -> position in y
    """
    dim = 2

    def __init__(self, sensor_bias, sigma_position, sigma_rangerate, state_space = 'CV'):
        """ Input:
                sigma_position: Standard deviation of zero mean Gaussian measurement noise.
                state_dim: dimensionality of the state vector.
                sensor_bias: E.g position of sensor. Numpy array of shape (state_dim,)
        """
        self.R = np.diag((sigma_position**2, sigma_rangerate**2))
        self.sensor_bias = sensor_bias.squeeze()
        self.state_space = state_space
        self.initialize_jacobian()


    def initialize_jacobian(self):
        if self.state_space == 'CV':
            # Evaluate Jacobian
            px, py, vx, vy = sym.symbols('px py vx vy')
            self._h = sym.Matrix([ sym.sqrt( (px - self.sensor_bias[0])**2 + (py - self.sensor_bias[1])**2 ),
                (vx*(px - self.sensor_bias[0]) + vy*(py - self.sensor_bias[1]))/sym.sqrt((px - self.sensor_bias[0])**2 + (py - self.sensor_bias[1])**2)])
            # Store jacobian and symbolic variables
            self.x = [px, py, vx, vy]
            self._H = self._h.jacobian([px, py, vx, vy])
    
        elif self.state_space == 'CT':
            # Evaluate Jacobian
            px, py, v, heading, turnrate = sym.symbols('px py v heading turnrate')
            self._h = sym.Matrix([ sym.sqrt( (px - self.sensor_bias[0])**2 + (py - self.sensor_bias[1])**2 ),
                (v*sym.cos(heading)*(px - self.sensor_bias[0]) + v*sym.sin(heading)*(py - self.sensor_bias[1]))/sym.sqrt((px - self.sensor_bias[0])**2 + (py - self.sensor_bias[1])**2)])
            # Store jacobian and symbolic variables
            self.x = [px, py, v, heading, turnrate]
            self._H = self._h.jacobian([px, py, v, heading, turnrate])

    def H(self, x):
        """ Returns the measurement model Jacobian. 
        """
        x = x.squeeze()
        H = self._H.subs(list(zip(self.x, x)))
        return np.array(H).astype(np.float64)
    
    def h(self, x):
        x = x.squeeze()
        h = self._h.subs(list(zip(self.x, x)))
        return np.array(h).astype(np.float64)


if __name__ == '__main__':
    meas_model = RangeDopplerMeasurmentModel(sensor_bias = np.array([1,1]), sigma_position = 1, sigma_rangerate = 1, state_space = 'CV')
    A = meas_model.H(np.array([1,2,3,4]))
