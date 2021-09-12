from copy import deepcopy
from math import e

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
import matplotlib.colors
from scipy import stats

import processmodels
import measmodels
import render
from density import Gaussian
import radar_configuration

def generate_ground_truth(process_model, initial_state, n_steps):
    """ Generates ground truth sequence.

    Parameters:
        instance of process_model: Dynamical model
        initial_state (np.array): Initial state
        T: time discritization
        n_steps (int): Number of discrete time steps
    """
    initial_state = np.reshape(initial_state, (np.size(initial_state),1))
    sequence = [initial_state]
    for t in range(n_steps-1):
        sequence.append(process_model.f(sequence[-1]))
    return np.hstack(sequence).T


def measurement_sequence(measurement_model, ground_truth, seed = None):
    """ Generates measurement sequence.

    Parameters:
        instance of measurement model:
        ground_truth: Ground truth sequence
    """
    measurements = []
    for state in ground_truth:
        measurements.append(multivariate_normal.rvs(measurement_model.h(state).squeeze(), measurement_model.R, random_state = seed))
    return np.vstack(measurements)


def sigma_bound(mu, sigma, level = 3, n_points = 32):
    """ Calculates the sigma bound elipse for a 2D Gaussian

    Parameters:
        mu (np.array of shape (2,1)): Mean vector
        sigma (np.array of shape (2,2)): Covariance vector
        level (int): sigma bound level
    
    Returns:
        elipse (np.array of size (n_points,2)): Uniform samples of the sigma bound
    """
    mu = np.reshape(mu, (mu.shape[0],1))

    linspace = np.linspace(0,2*np.pi, n_points)
    z = np.vstack([[level*np.cos(linspace)], [level*np.sin(linspace)]])
    elipse = mu + sqrtm(sigma) @ z
    return elipse.T


def ct_to_cv(state):
    state = state.squeeze()
    return np.array([[state[0]],[state[1]],[state[2]*np.cos(state[3])],[state[2]*np.cos(state[3])]])


def get_angle_estimates(state, sensor_positions, n=50000, n_bins = 4):
    state = deepcopy(state)
    if state.x[2] < 0:
        state.x[3] = state.x[3] + np.pi

    state_samples = multivariate_normal.rvs(state.x.squeeze(), state.P, size=n)
    estimates = []
    for sensor in sensor_positions:
        estimates.append(estimate_angle(state_samples, sensor, n_bins))
    return estimates


def estimate_angle(state_samples, sensor, n_bins):

    sensor_pos = sensor - state_samples[:, 0:2]
    state_samples[:, 3] = np.mod(state_samples[:, 3], 2*np.pi)

    # get angle from target to sensor.
    sensor_angle = np.mod(np.arctan2(sensor_pos[:,1], sensor_pos[:,0]), 2*np.pi) # Note that first argument is y axis

    # Angle between sensor and heading angle
    viewing_angle = np.mod(sensor_angle - state_samples[:, 3], 2*np.pi)

    # Discretization
    bin_size = np.pi/n_bins # bin resolution 2 times higher to enable wrapping bins around 0
    bins = [ bin_size * i for i in range(n_bins * 2 + 1) ]

    binned_values = list(np.digitize(viewing_angle, bins))
    counts = np.array([ binned_values.count(i+1) for i in range(n_bins*2) ])

    estimates = counts/np.sum(counts)
    estimates = np.roll(estimates, shift=1)
    estimates = np.array([ estimates[i] + estimates[i+1] for i in range(0, estimates.shape[0], 2) ])

    return estimates


def static_bayesian_fusion(df, n_classes):

    samples = df.groupby(['time', 'file_index'])
    for name, sample in samples:
        belief = np.array([1/n_classes for _ in range(n_classes)])
        for row_index, row in sample.iterrows():
            belief = belief*row['belief']
            belief = belief/(belief.sum())
            print('s')

        df.loc[sample.index,'prediction'] = belief.argmax() + 1
        for index in sample.index:
            df.at[index,'belief'] = belief

    return None


def pca_filter(spectrogram, pca):
    n_doppler_bins, n_time_bins = spectrogram.shape
    spectrogram = spectrogram.reshape(1, n_doppler_bins * n_time_bins).copy()
    principal_components = pca.transform(spectrogram)
    spectrogram = pca.inverse_transform(principal_components)
    spectrogram = spectrogram.reshape(n_doppler_bins, n_time_bins)
    return spectrogram


class InverseEntropy:
    def __init__(self):
        
        confidence = np.linspace(0, 1, num=10000)
        entropy = stats.entropy(np.stack((confidence, 1-confidence)), base=e, axis=0)
        self.inverse = np.stack((entropy, confidence)).T
    
    def inverse_entropy(self, entropy):
        index = np.abs(self.inverse[:,0] - entropy).argmin()
        return self.inverse[index,1]


if __name__ == '__main__':
    inv_entropy = InverseEntropy()
    inv_entropy.inverse_entropy(0.5)
    print('s')