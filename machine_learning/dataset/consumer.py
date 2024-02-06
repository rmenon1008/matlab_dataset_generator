import json
import os
import sys
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from cprint import *
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .path_utils import breesenham, curve_in_range


class DatasetBuilder:
    def __init__(self, dataset_path):
        self.attributes = None
        self.csi_mags = None
        self.csi_phases = None
        self.rx_positions = None
        self.ray_aoas = None

        with h5py.File(dataset_path, "r") as file:
            self.attributes = self.__clean_attributes(file.attrs)
            self.csi_mags = file["csis_mag"][:]
            self.csi_phases = file["csis_phase"][:]
            self.rx_positions = file["positions"][:]
            self.ray_aoas = file["ray_aoas"][:]

        self.tx_position = self.attributes["tx_position"]
        self.grid_size, self.grid_spacing = self.__find_grid(self.rx_positions)

    def __find_grid(self, rx_positions):
        # Find the grid size and spacing that was used to generate the dataset
        min_x = np.min(rx_positions[0, :])
        max_x = np.max(rx_positions[0, :])
        min_y = np.min(rx_positions[1, :])
        max_y = np.max(rx_positions[1, :])
        grid_bounds = (min_x, max_x, min_y, max_y)

        grid_spacing = self.attributes["rx_grid_spacing"]

        return grid_bounds, grid_spacing

    def __real_to_grid(self, x, y):
        # Find the index in the grid that the given point is in
        # Return None if the point is not in the grid
        if x < self.grid_size[0] or x > self.grid_size[1]:
            return None
        if y < self.grid_size[2] or y > self.grid_size[3]:
            return None

        x_index = int((x - self.grid_size[0]) / self.grid_spacing)
        y_index = int((y - self.grid_size[2]) / self.grid_spacing)

        return x_index, y_index

    def __grid_to_real_index(self, x, y):
        # Find the real position of the given grid index
        x_real = self.grid_size[0] + (x * self.grid_spacing)
        y_real = self.grid_size[2] + (y * self.grid_spacing)

        # Find the closest point in the rx_positions array
        index = np.argmin(
            np.abs(self.rx_positions[0, :] - x_real)
            + np.abs(self.rx_positions[1, :] - y_real)
        )
        return index

    def __closest_real_index(self, x, y):
        # Find the closest point in the rx_positions array
        index = np.argmin(
            np.abs(self.rx_positions[0, :] - x) + np.abs(self.rx_positions[1, :] - y)
        )
        return index

    def __clean_attributes(self, attributes):
        def replace_np_array_with_list(d):
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    if v.size == 1:
                        d[k] = v.item()
                    else:
                        d[k] = v.tolist()
                elif isinstance(v, dict):
                    replace_np_array_with_list(v)

        d = dict(attributes)
        replace_np_array_with_list(d)
        return d

    def __single_straight_path(self, start_point, end_point):
        # Generate a straight path between two points
        x0, y0 = self.__real_to_grid(start_point[0], start_point[1])
        x1, y1 = self.__real_to_grid(end_point[0], end_point[1])
        points = breesenham(x0, y0, x1, y1)
        return np.array([self.__grid_to_real_index(x, y) for x, y in points])

    def print_info(self):
        print(json.dumps(self.attributes, indent=2))

    def __generate_straight_paths(self, num_paths, path_length_n):
        """
        Generate straight paths in the rx_positions array.

        num_paths: Number of paths to generate
        path_length_n: Length of each path in number of points
        """
        print(f"Generating {num_paths} paths of length {path_length_n}")

        path_indices = np.zeros((num_paths, path_length_n), dtype=np.int32)
        for i in range(num_paths):
            while True:
                # Grab two random points
                point1 = np.random.randint(0, self.rx_positions.shape[1])
                point2 = np.random.randint(0, self.rx_positions.shape[1])

                # Generate a straight line between the two points
                path = self.__single_straight_path(
                    self.rx_positions[:, point1], self.rx_positions[:, point2]
                )

                # Make sure the path is at least path_length_n long
                if len(path) <= path_length_n:
                    continue

                path_indices[i] = path[:path_length_n]
                break

        return path_indices

    def __generate_curved_paths(self, num_paths, path_length_n=20):
        """
        Generate curved paths in the rx_positions array. (currently not working as expected)

        inputs:
            num_paths: Number of paths to generate
            path_length_n: Length of each path in number of points
        output: An array with shape: (num_paths, path_length_n)
        """
        deg2rad = np.pi / 180.0
        rad2deg = 180.0 / np.pi
        print(f"Generating {num_paths} paths of length {path_length_n}")
        path_indices = np.zeros((num_paths, path_length_n), dtype=np.int32)
        for i in range(num_paths):
            while True:
                # Grab one random point within the size of the dataset
                point1 = np.random.randint(0, self.rx_positions.shape[1])
                # reducing randomness by setting 2 degrees
                steering_angle_delta = np.random.uniform(-1, 1)
                heading_angle_theta = np.random.uniform(0, 360)

                # angles in radians
                delta = steering_angle_delta * deg2rad
                theta = heading_angle_theta * deg2rad
                dt = 1  # this value can be changed, currently the time step is 1

                # Generate a curved line between from random point and set angle
                x, y = self.__real_to_grid(
                    self.rx_positions[0, point1], self.rx_positions[1, point1]
                )
                range_of_points = np.arange(0, path_length_n, dt)
                [x_coor, y_coor, points, thetas] = curve_in_range(
                    delta, 1, theta, x, y, range_of_points, 0.2, dt
                )

                # Check if all the points are within the grid bounds
                ep = 10
                if np.any(np.array(x_coor) < self.grid_size[0] + ep) or np.any(
                    np.array(x_coor) > self.grid_size[1] - ep
                ):
                    continue
                if np.any(np.array(y_coor) < self.grid_size[2] + ep) or np.any(
                    np.array(y_coor) > self.grid_size[3] - ep
                ):
                    continue

                # Make sure the path is at least path_length_n long
                path_indices[i] = [
                    self.__grid_to_real_index(x, y) for x, y in points[:path_length_n]
                ]
                break

        return path_indices

    def __create_terminal(self, path_indices, dir, terminal_length):
        def rotate_vec(vec, angle):
            """
            Rotate a vector by the given angle
            """
            flat = np.array([vec[0], vec[1]])
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            angled = np.matmul(rotation_matrix, flat)
            return np.array([angled[0], angled[1], vec[2]])

        num_paths, path_length_n = path_indices.shape
        left_paths = np.zeros(
            (num_paths, path_length_n + terminal_length), dtype=np.int32
        )
        center_paths = np.zeros(
            (num_paths, path_length_n + terminal_length), dtype=np.int32
        )
        right_paths = np.zeros(
            (num_paths, path_length_n + terminal_length), dtype=np.int32
        )
        for i in range(num_paths):
            # Add the base path
            left_paths[i, :path_length_n] = path_indices[i]
            center_paths[i, :path_length_n] = path_indices[i]
            right_paths[i, :path_length_n] = path_indices[i]

            # Find the direction at the end of the path
            last_point = self.rx_positions[:, path_indices[i, -1]]
            second_to_last_point = self.rx_positions[:, path_indices[i, -5]]
            direction = last_point - second_to_last_point

            if np.linalg.norm(direction) == 0:
                direction = np.array([1, 0, 0])
            else:
                direction = direction / np.linalg.norm(direction)

            # Determine the direction for the terminal
            if dir == "left":
                terminal_direction = rotate_vec(direction, np.pi / 4)
            elif dir == "center":
                terminal_direction = direction
            elif dir == "right":
                terminal_direction = rotate_vec(direction, -np.pi / 4)
            else:
                raise ValueError(
                    "Invalid direction. Choose 'left', 'center', or 'right'."
                )

            # Find the end point of the terminal
            terminal_end = (
                last_point
                + terminal_direction * terminal_length * self.grid_spacing * 2
            )

            # Find the index of the end point
            terminal_index = self.__closest_real_index(terminal_end[0], terminal_end[1])

            # Create a straight path to the end point
            terminal_path = self.__single_straight_path(
                last_point, self.rx_positions[:, terminal_index]
            )

            # Pad or trim the path as necessary
            def pad_path(path):
                if len(path) < terminal_length:
                    path = np.repeat(path[-1:], terminal_length, axis=0)
                if len(path) > terminal_length:
                    path = path[:terminal_length]
                return path

            terminal_path = pad_path(terminal_path)

            # Create the new path array
            new_path = np.zeros(
                (num_paths, path_length_n + terminal_length), dtype=np.int32
            )
            for i in range(num_paths):
                new_path[i, :path_length_n] = path_indices[i]
                new_path[i, path_length_n:] = terminal_path

            return new_path

    def __relative_pos_section(self, path_indices):
        """
        Generate a dataset from the given path indices
        Shape: (num_paths, path_length_n, 2)
        """
        # Use the indices to grab the positions for each point
        positions = self.rx_positions[:, path_indices]

        # Subtract the starting point from all points
        positions = positions - positions[:, :, 0:1]

        # Divide by the grid spacing
        positions = positions / self.grid_size[1]

        # Remove the z axis
        positions = np.swapaxes(positions, 0, 1)
        positions = np.swapaxes(positions, 1, 2)
        return positions[:, :, 0:2]

    def __mag_section(self, path_indices):
        csi_mags = self.csi_mags[:, path_indices]
        csi_mags = np.swapaxes(csi_mags, 0, 1)
        csi_mags = np.swapaxes(csi_mags, 1, 2)
        return csi_mags

    def __phase_section(self, path_indices):
        csi_phases = self.csi_phases[:, path_indices]
        csi_phases = np.swapaxes(csi_phases, 0, 1)
        csi_phases = np.swapaxes(csi_phases, 1, 2)
        return csi_phases

    def __mag_phase_interleaved_section(self, path_indices):
        csi_mags = self.__mag_section(path_indices)
        csi_phases = self.__phase_section(path_indices)
        interleaved = np.zeros(
            (
                csi_mags.shape[0],
                csi_mags.shape[1],
                csi_mags.shape[2] + csi_phases.shape[2],
            )
        )
        interleaved[:, :, ::2] = csi_mags
        interleaved[:, :, 1::2] = csi_phases
        return interleaved

    def __real_section(self, path_indices):
        csi_mags = self.csi_mags[:, path_indices]
        csi_phases = self.csi_phases[:, path_indices]
        csi_real = csi_mags * np.cos(csi_phases)
        csi_real = np.swapaxes(csi_real, 0, 1)
        csi_real = np.swapaxes(csi_real, 1, 2)
        return csi_real

    def __imag_section(self, path_indices):
        csi_mags = self.csi_mags[:, path_indices]
        csi_phases = self.csi_phases[:, path_indices]
        csi_imag = csi_mags * np.sin(csi_phases)
        csi_imag = np.swapaxes(csi_imag, 0, 1)
        csi_imag = np.swapaxes(csi_imag, 1, 2)
        return csi_imag

    def __real_imag_interleaved_section(self, path_indices):
        csi_real = self.__real_section(path_indices)
        csi_imag = self.__imag_section(path_indices)
        interleaved = np.zeros(
            (
                csi_real.shape[0],
                csi_real.shape[1],
                csi_real.shape[2] + csi_imag.shape[2],
            )
        )
        interleaved[:, :, ::2] = csi_real
        interleaved[:, :, 1::2] = csi_imag
        return interleaved

    def generate_dataset(self, options):
        """
        Generate a dataset from the given options.
        """

        # Handle scaling the magnitude data
        match options["mag_scaler"]:
            case "maxabs":
                scaler = MaxAbsScaler()
            case "minmax":
                scaler = MinMaxScaler()
            case "power":
                scaler = PowerTransformer()
            case "quantile":
                scaler = QuantileTransformer()
            case "robust":
                scaler = RobustScaler()
            case "standard":
                scaler = StandardScaler()
            case "none":
                scaler = None
            case _:
                raise ValueError(f"Invalid scaler: {options['mag_scaler']}")

        if scaler:
            scaler.fit(self.csi_mags.T)
            self.csi_mags = scaler.transform(self.csi_mags.T).T

        # Generate the paths
        match options["paths"]["path_type"]:
            case "straight":
                path_indices = self.__generate_straight_paths(
                    options["n_paths"], options["paths"]["path_length"]
                )
            case "curved":
                path_indices = self.__generate_curved_paths(
                    options["n_paths"], options["paths"]["path_length"]
                )
            case _:
                raise ValueError(f"Invalid path_type: {options['paths']['path_type']}")

        # Handle the terminals
        if (
            options["paths"]["terminal_length"] is not None
            and options["paths"]["terminal_length"] > 0
        ):
            path_indices = self.__create_terminal(
                path_indices,
                options["paths"]["terminal_direction"],
                options["paths"]["terminal_length"],
            )

        # Create the dataset
        sections = []
        section_widths = []
        for section in options["dataset_sections"]:
            match section:
                case "mag":
                    section = self.__mag_section(path_indices)
                    sections.append(section)
                    section_widths.append(section.shape[2])
                case "phase":
                    section = self.__phase_section(path_indices)
                    sections.append(section)
                    section_widths.append(section.shape[2])
                case "mag_phase_interleaved":
                    section = self.__mag_phase_interleaved_section(path_indices)
                    sections.append(section)
                    section_widths.append(section.shape[2])
                case "real":
                    section = self.__real_section(path_indices)
                    sections.append(section)
                    section_widths.append(section.shape[2])
                case "imag":
                    section = self.__imag_section(path_indices)
                    sections.append(section)
                    section_widths.append(section.shape[2])
                case "real_imag_interleaved":
                    section = self.__real_imag_interleaved_section(path_indices)
                    sections.append(section)
                    section_widths.append(section.shape[2])
                case "relative_position":
                    section = self.__relative_pos_section(path_indices)
                    sections.append(section)
                    section_widths.append(section.shape[2])
                case _:
                    raise ValueError(f"Invalid dataset_section: {section}")
        dataset = np.concatenate(sections, axis=2)

        def path_point_to_csi(point):
            print(f"point: {point.shape}")
            csi_mag = None
            csi_phase = None
            if "mag" in options["dataset_sections"]:
                mag_section = options["dataset_sections"].index("mag")
                mag_start_index = sum(section_widths[:mag_section])
                mag_end_index = mag_start_index + section_widths[mag_section]
                csi_mag = point[mag_start_index:mag_end_index]

            if "phase" in options["dataset_sections"]:
                phase_section = options["dataset_sections"].index("phase")
                phase_start_index = sum(section_widths[:phase_section])
                phase_end_index = phase_start_index + section_widths[phase_section]
                csi_phase = point[phase_start_index:phase_end_index]

            if "mag_phase_interleaved" in options["dataset_sections"]:
                mag_phase_section = options["dataset_sections"].index(
                    "mag_phase_interleaved"
                )
                mag_phase_start_index = sum(section_widths[:mag_phase_section])
                mag_phase_end_index = (
                    mag_phase_start_index + section_widths[mag_phase_section]
                )
                csi_mag = point[mag_phase_start_index:mag_phase_end_index:2]
                csi_phase = point[mag_phase_start_index + 1 : mag_phase_end_index : 2]

            if (
                "real" in options["dataset_sections"]
                and "imag" in options["dataset_sections"]
            ):
                real_section = options["dataset_sections"].index("real")
                imag_section = options["dataset_sections"].index("imag")
                real_start_index = sum(section_widths[:real_section])
                real_end_index = real_start_index + section_widths[real_section]
                imag_start_index = sum(section_widths[:imag_section])
                imag_end_index = imag_start_index + section_widths[imag_section]

                print(f"real_start_index: {real_start_index}")
                print(f"real_end_index: {real_end_index}")
                print(f"imag_start_index: {imag_start_index}")
                print(f"imag_end_index: {imag_end_index}")

                csi_real = point[real_start_index:real_end_index]
                csi_imag = point[imag_start_index:imag_end_index]

                print(f"csi_real: {csi_real.shape}")
                print(f"csi_imag: {csi_imag.shape}")

                csi_mag = np.sqrt(csi_real**2 + csi_imag**2)
                csi_phase = np.arctan2(csi_imag, csi_real)

                print(f"csi_mag: {csi_mag.shape}")
                print(f"csi_phase: {csi_phase.shape}")

            if "real_imag_interleaved" in options["dataset_sections"]:
                real_imag_section = options["dataset_sections"].index(
                    "real_imag_interleaved"
                )
                real_imag_start_index = sum(section_widths[:real_imag_section])
                real_imag_end_index = (
                    real_imag_start_index + section_widths[real_imag_section]
                )
                csi_real = point[real_imag_start_index:real_imag_end_index:2]
                csi_imag = point[real_imag_start_index + 1 : real_imag_end_index : 2]
                csi_mag = np.sqrt(csi_real**2 + csi_imag**2)
                csi_phase = np.arctan2(csi_imag, csi_real)

            print(f"csi_mag: {csi_mag.shape}")
            print(f"csi_phase: {csi_phase.shape}")

            return scaler.inverse_transform(csi_mag.reshape(1, -1)).squeeze(), csi_phase

        return dataset, path_point_to_csi


# def plot_csi(mag, phase):
#     plt.figure(figsize=(20, 10))
#     plt.subplot(2, 1, 1)
#     plt.plot(mag)
#     plt.title("Magnitude")
#     plt.subplot(2, 1, 2)
#     plt.plot(phase)
#     plt.title("Phase")
#     plt.show()


# DATASET = "dataset_0_5m_spacing.h5"
# d = DatasetBuilder(DATASET)
# d.print_info()

# options = {
#     "n_paths": 1000,
#     "paths": {
#         "path_type": "curved",
#         "path_length": 20,
#         "terminal_length": 5,
#         "terminal_direction": "center",
#     },
#     "mag_scaler": "quantile",
#     "dataset_sections": ["real_imag_interleaved", "relative_position"],
# }

# dataset, inverse_transform = d.generate_dataset(options)
# mag, phase = inverse_transform(dataset[0][0])
# plot_csi(mag, phase)
