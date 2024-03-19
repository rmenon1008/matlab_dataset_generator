import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
from typing import Callable
from cprint import *
from dataset_consumer import DatasetConsumer

DATASET = './machine_learning/data/dataset_0_5m_spacing.h5'
d = DatasetConsumer(DATASET)
# d.print_info()

mags = d.csi_mags
aoas = d.ray_aoas
mags_fromloss = d.ray_path_losses

# Graph 1 - Plotting the first set of rays magnitude & aoa ray pairs for positions 0 to 10 
# EXAMPLE: mags_1 = mags[0,:10] # first set of rays, magnitudes at first 10 positions
mags_1 = mags_fromloss[0, :10]
print(mags_1)

aoas_1 = np.deg2rad(aoas[0,0,:10]) # first set of rays, azimuth, first 10 positions
print(np.rad2deg(aoas_1))

# Plot in polar coordinates, changing min and max of radial coordinates based on magnitudes
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
scatter = ax.scatter(aoas_1, mags_1, c=mags_1, cmap='viridis') # plotting the polar coordinates as a scatter plot
ax.set_rmin(np.min(mags_1))
ax.set_rmax(np.max(mags_1) + 1)

# Add colorbar for reference - purple to yellow, purple is lower magnitude
cbar = plt.colorbar(scatter, ax=ax, label='Magnitude Color Reference')
plt.savefig('plot_mags_aoas/first10positions_ray0_fig_1.png')

# Graph 2 - Plotting aoa for first 10 rays at position 0 with path loss as the magnitude
mags_2 = mags_fromloss[:10, 0]
print(mags_2)

aoas_2 = np.deg2rad(aoas[:10,0,0]) # first set of rays, azimuth, first 10 positions
print(np.rad2deg(aoas_2))

# Plot in polar coordinates, changing min and max of radial coordinates based on magnitude range
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
scatter = ax.scatter(aoas_2, mags_2, c=mags_2, cmap='viridis') # plotting the polar coordinates as a scatter plot
ax.set_rmin(np.min(mags_2))
ax.set_rmax(np.max(mags_2) + 1)

# Add colorbar for reference - purple to yellow, purple is lower magnitude
cbar = plt.colorbar(scatter, ax=ax, label='Magnitude Color Reference')
plt.savefig('plot_mags_aoas/first10rays_position0_fig_2.png')

# Graph 3 - Generate ONE straight path and plot the points on it - note that each position can have up
#           to 100 rays but will likely be less. Each point represents the magnitude and aoa of a ray.

num_points = 5 # number of points on path, indicates the different positions
paths = d.generate_straight_paths(1, num_points)

mags_3 = d.paths_to_dataset_path_loss_only(paths)
print(mags_3.shape) # Shape: (1,num_points,100) -> 1 path, num_points, up to 100 rays per point/position

aoas_3 = d.paths_to_dataset_rays_aoas(paths)[0]
print(aoas_3)

# Setting up plot of polar coordinates, width of saved figure will change based on number of points
fig, ax = plt.subplots(1, num_points, subplot_kw=dict(projection="polar"),figsize=(28, num_points*3))
# ax.set_rmin(np.min(mags_3[0,1,:]))
# ax.set_rmax(np.max(mags_3[0,1,:]) + 1)
for n in range(num_points):
    random_colormap = np.random.choice([cm.Purples, cm.Greens, cm.Reds, cm.Blues, cm.Oranges])
    # Create a polar graph with all 100 rays (may have fewer) for n'th position
    scatter = ax[n].scatter(aoas_3[0,n,:], mags_3[0,n,:], c=mags_3[0,n,:], cmap=random_colormap) # plotting the polar coordinates as a scatter plot
    # Add colorbar for reference beside each point graph
    cbar = plt.colorbar(scatter, ax=ax[n], label='Magnitude Color Reference (Point %d)'% n, shrink=0.8)
    ax.set_rmin(n, np.min(mags_3[0,n,:]))
    # ax.set_rmax(1, n, np.max(mags_3[0,n,:]) + 1)
    # ax.set_rlim(1, n, np.min(mags_3), np.max(mags_3) + 1)
# adding padding between graphs
plt.subplots_adjust(wspace = 1)
plt.savefig('plot_mags_aoas/straight_line_aoas+mags_per_point.png')





