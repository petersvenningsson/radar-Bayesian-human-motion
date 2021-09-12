import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats.distributions import chi2

class Gaussian():
    """ Collection of static mehods related to operations on
    Gaussian distributions.
    """

    @classmethod
    def predict(cls, state, process_model):
        predicted_mean = process_model.f(state.x)
        predicted_covariance = process_model.F(state.x) @ state.P @ \
            process_model.F(state.x).T + process_model.Q

        return cls(predicted_mean, predicted_covariance)


    @classmethod
    def update(cls, state, measurement, measurement_model, linearization = None):
        if linearization is None:
            linearization = state.x

        measurement = np.reshape(measurement, (measurement.shape[0], 1))

        H = measurement_model.H(linearization)
        S = H @ state.P @ H.T + measurement_model.R
        S = (S+S.T)/2 # Enforce positive definte
        K = (state.P @ H.T) @ np.linalg.pinv(S)

        updated_mean = state.x + K @ (measurement - measurement_model.h(state.x))
        updated_covariance = (np.identity(state.x.shape[0]) - K @ H) @ state.P 

        return cls(updated_mean, updated_covariance)


    @classmethod
    def moment_matching(cls, states, weights, log_scale = True):
        for state in states:
            if len(state.x.shape) == 1:
                state.x = state.x[:, np.newaxis]

        if log_scale:
            weights = np.exp(weights)

        matched_mean = np.sum(tuple(( s.x * w for s,w in zip(states, weights))), 0)
    
        cov_generator = [w*( s.P  + (s.x - matched_mean) @ (s.x - matched_mean).T ) \
            for s,w in zip(states, weights)]

        matched_covariance = np.add(*tuple((cov_generator)))
        
        return cls(matched_mean, matched_covariance)


    @staticmethod
    def likelihood(state, measurement, measurement_model, log_scale = True):
        
        H = measurement_model.H(state.x)
        S = H @ state.x @ H.T + measurement.R
        S = (S+S.T)/2 # Enforce positive definite
        mean = measurement_model.h(state.x)
        
        if log_scale:
            return np.log(multivariate_normal.pdf(measurement, mean, S))
        else:
            return multivariate_normal.pdf(measurement, mean, S)


    @staticmethod
    def ellipsoidalGating(measurement, state, measurement_model, gating_probability = 0.80):
        if measurement is None:
            return None
        measurement = np.reshape(measurement, (measurement.shape[0], 1))

        gating_size = chi2.ppf(gating_probability, df=measurement_model.dim)
        H = measurement_model.H(state.x)
        S = H @ state.P @ H.T + measurement_model.R
        S = (S+S.T)/2 # Enforce positive definte

        mean = measurement_model.h(state.x)
        distance_sq = (measurement - mean).T @ np.linalg.pinv(S) @ (measurement - mean)

        if distance_sq < gating_size:
            return measurement
        else:
            print('Removed gated measurement')
            return None


    def __init__(self, mean, covariance):
        self.x = mean
        self.P = covariance

    @property
    def mean(self):
        return self.x


    @property
    def covariance(self):
        return self.P


if __name__ == '__main__':
    A = Gaussian(1,2)
    B = Gaussian(3,4)
    C = Gaussian.moment_matching([A, B], [1,2] )