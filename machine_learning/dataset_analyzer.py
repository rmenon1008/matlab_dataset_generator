# Author:  Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
# License: BSD 3 clause

import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from dataset_consumer import DatasetConsumer

from cprint import *

from utils import watts_to_dbm, get_scaler


SCALER = "power_box-cox"
DATASET = 'dataset_0_5m_spacing.h5'
d = DatasetConsumer(DATASET)
d.print_info()

# Mags
# d.csi_mags = d.scale(scaler.fit_transform, d.csi_mags.T).T
not_scaled = d.csi_mags
d.csi_mags = watts_to_dbm(d.csi_mags)
scaler = get_scaler(SCALER)
scaler.fit(d.csi_mags.T)
d.csi_mags = d.scale(scaler.transform, d.csi_mags.T).T

# Phase
d.csi_phases = d.unwrap(d.csi_phases)
scaler_phase = get_scaler('minmax')
d.csi_phases = d.scale(scaler.fit_transform, d.csi_phases)


# X_full, y_full = dataset.data, dataset.target
# feature_names = dataset.feature_names

# feature_mapping = {
#     "MedInc": "Median income in block",
#     "HouseAge": "Median house age in block",
#     "AveRooms": "Average number of rooms",
#     "AveBedrms": "Average number of bedrooms",
#     "Population": "Block population",
#     "AveOccup": "Average house occupancy",
#     "Latitude": "House block latitude",
#     "Longitude": "House block longitude",
# }

# # Take only 2 features to make visualization easier
# # Feature MedInc has a long tail distribution.
# # Feature AveOccup has a few but very large outliers.
# features = ["MedInc", "AveOccup"]
# features_idx = [feature_names.index(feature) for feature in features]

# X = d.csi_mags
# distributions = [
#     ("Unscaled data", X),
#     ("Data after standard scaling", StandardScaler().fit_transform(X)),
#     ("Data after min-max scaling", MinMaxScaler().fit_transform(X)),
#     ("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),
#     (
#         "Data after robust scaling",
#         RobustScaler(quantile_range=(25, 75)).fit_transform(X),
#     ),
#     (
#         "Data after power transformation (Yeo-Johnson)",
#         PowerTransformer(method="yeo-johnson").fit_transform(X),
#     ),
#     (
#         "Data after power transformation (Box-Cox)",
#         PowerTransformer(method="box-cox").fit_transform(X),
#     ),
#     (
#         "Data after quantile transformation (uniform pdf)",
#         QuantileTransformer(
#             output_distribution="uniform", random_state=42
#         ).fit_transform(X),
#     ),
#     (
#         "Data after quantile transformation (gaussian pdf)",
#         QuantileTransformer(
#             output_distribution="normal", random_state=42
#         ).fit_transform(X),
#     ),
#     ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X)),
# ]

# scale the output between 0 and 1 for the colorbar
# y = minmax_scale(np.zeros(100)) #y_full)

# plasma does not exist in matplotlib < 1.5
# cmap = getattr(cm, "plasma_r", cm.hot_r)


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )


def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cmap(y)
    # ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)
    ax.scatter(X[:], np.arange(X.shape[1]), alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # # Removing the top and the right spine for aesthetics
    # # make nice axis layout
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # ax.spines["left"].set_position(("outward", 10))
    # ax.spines["bottom"].set_position(("outward", 10))

    # # Histogram for axis X1 (feature 5)
    # hist_X1.set_ylim(ax.get_ylim())
    # hist_X1.hist(
    #     X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    # )
    # hist_X1.axis("off")

    # # Histogram for axis X0 (feature 0)
    # hist_X0.set_xlim(ax.get_xlim())
    # hist_X0.hist(
    #     X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    # )
    # hist_X0.axis("off")

def make_plot(item_idx):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(
        axarr[0],
        X,
        y,
        hist_nbins=200,
        x0_label='test', #feature_mapping[features[0]],
        x1_label='test', #feature_mapping[features[1]],
        title="Full data",
    )

    # # zoom-in
    # zoom_in_percentile_range = (0, 99)
    # cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    # cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    # non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
    #     X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
    # )
    # plot_distribution(
    #     axarr[1],
    #     X[non_outliers_mask],
    #     y[non_outliers_mask],
    #     hist_nbins=50,
    #     x0_label='test', #feature_mapping[features[0]],
    #     x1_label='test', #feature_mapping[features[1]],
    #     title="Zoom-in",
    # )

    # norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    # mpl.colorbar.ColorbarBase(
    #     ax_colorbar,
    #     cmap=cmap,
    #     norm=norm,
    #     orientation="vertical",
    #     label="Color mapping for values of y",
    # )

# make_plot(0)

# data = d.csi_mags[:,:1]

# # Plot the 2D array using imshow
# plt.imshow(data, cmap='viridis', aspect='auto', origin='lower')  # Choose colormap, adjust aspect ratio, and set origin

# # Add colorbar for reference
# plt.colorbar(label='Colorbar Label')

# # Set labels for x and y axes
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')

# # Show the plot
# plt.show()

positions = np.sqrt((d.rx_positions[0,:] - 100) ** 2 + (d.rx_positions[1,:] - 170) ** 2)
positions = np.repeat(positions, 128)

hist_nbins = 500
ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(f'Data after {SCALER} scaling')
axarr = (ax_zoom_out, ax_zoom_in)
ax_0, hist_X1_0, hist_X0_0 = axarr[0]
ax_1, hist_X1_1, hist_X0_1 = axarr[1]

norm = mpl.colors.Normalize(np.min(positions), np.max(positions))
mpl.colorbar.ColorbarBase(
    ax_colorbar,
    # cmap=cmap,
    norm=norm,
    orientation="vertical",
    label="Color mapping for distance to TX",
)
cmap = getattr(cm, "plasma_r", cm.hot_r)
colors = cmap(positions)

# x: 50 to 150
# y: 120 to 220

ax_0.scatter(np.tile(np.arange(128), 40401), not_scaled.T.flatten(), s=0.25, marker=".", c=positions, cmap='viridis', alpha=0.25)
ax_0.set_title("CSI Mag")
ax_0.set_xlabel('Subcarrier')
ax_0.set_ylabel('CSI Mag (Linear)')
ax_1.scatter(np.tile(np.arange(128), 40401), d.csi_mags.T.flatten(), s=0.25, marker=".", c=positions, cmap='viridis', alpha=0.25)
ax_1.set_title("Scaled CSI Mag")
ax_1.set_xlabel('Subcarrier')
ax_1.set_ylabel('CSI Mag (Log and Scaled)')

# Histogram for axis X1 (feature 1)
# hist_X1_0.set_xlim(ax_0.get_xlim())
hist_X1_0.hist(
    not_scaled.T.flatten(), bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
)
hist_X1_0.axis("off")
hist_X0_0.set_visible(False)

# Histogram for axis X1 (feature 1)
# hist_X1_1.set_xlim(ax_1.get_xlim())
hist_X1_1.hist(
    d.csi_mags.T.flatten(), bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
)
hist_X1_1.axis("off")
hist_X0_1.set_visible(False)

plt.show()