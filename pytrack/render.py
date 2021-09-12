import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors

from . import processmodels
from . import utils

def render_ground_truth(*args, **kwargs):
    
    if not 'colormap' in kwargs:
        kwargs['colormap'] = plt.cm.Blues

    return render_quiver(*args, **kwargs)


def render_estimates_quiver(*args, **kwargs):
    
    if not 'colormap' in kwargs:
        kwargs['colormap'] = plt.cm.Greens

    # Extract state estimates from state objects
    args = list(args)
    states = np.hstack([ s['updated_state'].x for s in args[1]]).T
    args[1] = states
    return render_quiver(*tuple(args), **kwargs)


def render_quiver(figure, points, process_model, arrow_scale = 20, colormap = plt.cm.Blues):
    n_points = points.shape[0]
    colors = np.linspace(0.5, 0.8, n_points)
    if len(points) == 1:
        colors = [0.8]

    cmap = matplotlib.colors.ListedColormap(colormap(np.linspace(0.4,1,10)))

    if isinstance(process_model, processmodels.CoordinatedTurnModel):
        plt.quiver(points[:,0], points[:,1], points[:,2]*np.cos(points[:,3]), points[:,2]*np.sin(points[:,3]), 
            colors, cmap = cmap, width = 0.005, figure = figure, minlength = 0, scale = arrow_scale, minshaft = 0.00001)
    elif isinstance(process_model, processmodels.ConstantVelocityModel):
        plt.quiver(points[:,0], points[:,1], points[:,2], points[:,3], 
            colors, cmap = cmap, width = 0.005, figure = figure, minlength = 0, scale = arrow_scale, minshaft = 0.00001)
    
    plt.scatter(points[:,0], points[:,1], c = colors, s = 10, cmap = cmap, marker = 'o', figure = figure)
    
    return figure


def base_figure():
    figure = plt.figure(figsize=(12,8))
    plt.axis('equal')
    return figure


def render_measurements(figure, measurements, sensor_bias, color, marker = 'x'):

    plt.plot(measurements[:,0] + sensor_bias[0], measurements[:,1] + sensor_bias[1], marker, color = color, figure = figure)
    return figure


def render_estimate(figure, states, render_cov = True, markersize=3):
    if not states:
        return figure
    means = np.hstack([s['updated_state'].x for s in states]).T
    covariances = np.stack([s['updated_state'].covariance for s in states])
    n_points = means.shape[0]
    
    colors = np.linspace(0.3, 1, n_points)
    if len(colors) == 1:
        colors[0] = 1

    cmap = matplotlib.colors.ListedColormap(plt.cm.Greys(np.linspace(0.4,1,10)))
    for j, (mean, cov) in enumerate(zip(means, covariances)):
        plt.plot(mean[0], mean[1], c = cmap(colors[j]), markersize = markersize, marker = 'o')
        if render_cov:
            elipse = utils.sigma_bound(mean[0:2], cov[0:2,0:2], level = 3, n_points = 32)
            plt.plot(elipse[:,0], elipse[:,1], 'k--', figure=figure, color = cmap(colors[j]), linewidth=1)
    return figure