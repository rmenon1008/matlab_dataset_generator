import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
from typing import Callable

def breesenham(x0, y0, x1, y1):
    """
    Generate a line between two points using Breesenham's algorithm
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    points = []
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

class DatasetConsumer:
    def __init__(self, dataset_path):
        self.attributes = None
        self.csi_mags = None
        self.csi_phases = None
        self.rx_positions = None

        with h5py.File(dataset_path, 'r') as file:
            self.attributes = self.__clean_attributes(file.attrs)
            self.csi_mags = file['csis_mag'][:]
            self.csi_phases = file['csis_phase'][:]
            self.rx_positions = file['positions'][:]

        self.tx_position = self.attributes['tx_position']
        self.grid_size, self.grid_spacing = self.__find_grid(self.rx_positions)

    def __find_grid(self, rx_positions):
        # Find the grid size and spacing that was used to generate the dataset
        
        min_x = np.min(rx_positions[0, :])
        max_x = np.max(rx_positions[0, :])
        min_y = np.min(rx_positions[1, :])
        max_y = np.max(rx_positions[1, :])
        grid_bounds = (min_x, max_x, min_y, max_y)

        grid_spacing = self.attributes['rx_grid_spacing']

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
        index = np.argmin(np.abs(self.rx_positions[0, :] - x_real) + np.abs(self.rx_positions[1, :] - y_real))
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
    
    # Scale data with passed function
    def scale(self, scaler: Callable[[np.array], np.array], data: np.array) -> np.array:
        return scaler(data)
    
    # Unwrap data with passed function
    def unwrap(self, data: np.array, axis:int=0) -> np.array:
        return np.unwrap(data, axis=axis)

    def print_info(self):
        print(json.dumps(self.attributes, indent=2))
        

    def generate_straight_paths(self, num_paths, path_length_n=20):
        """
        Generate straight paths in the rx_positions array.

        num_paths: Number of paths to generate
        path_length_n: Length of each path in number of points
        """
        print(f'Generating {num_paths} paths of length {path_length_n}')
        
        path_indices = np.zeros((num_paths, path_length_n), dtype=np.int32)
        for i in range(num_paths):
            while True:
                # Grab two random points
                point1 = np.random.randint(0, self.rx_positions.shape[1])
                point2 = np.random.randint(0, self.rx_positions.shape[1])

                # Generate a straight line between the two points
                x0, y0 = self.__real_to_grid(self.rx_positions[0, point1], self.rx_positions[1, point1])
                x1, y1 = self.__real_to_grid(self.rx_positions[0, point2], self.rx_positions[1, point2])
                points = breesenham(x0, y0, x1, y1)

                # Make sure the path is at least path_length_n long
                if len(points) < path_length_n:
                    continue

                path_indices[i] = [self.__grid_to_real_index(x, y) for x, y in points[:path_length_n]]
                break
        return path_indices
    

    def paths_to_dataset_mag_only(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 128)
        """
        # Use the indices to grab the CSI data for each point
        csi_mags = self.csi_mags[:, path_indices]
        csi_mags = np.swapaxes(csi_mags, 0, 1)
        csi_mags = np.swapaxes(csi_mags, 1, 2)
        return csi_mags
    
    def paths_to_dataset_phase_only(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 128)
        """
        # Use the indices to grab the CSI data for each point
        csi_phases = self.csi_phases[:, path_indices]
        csi_phases = np.swapaxes(csi_phases, 0, 1)
        csi_phases = np.swapaxes(csi_phases, 1, 2)
        return csi_phases
    
    def paths_to_dataset_positions(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 2)
        """
        # Use the indices to grab the positions for each point
        positions = self.rx_positions[:, path_indices]
        positions = np.swapaxes(positions, 0, 1)
        return positions
        

# DATASET = 'dataset_0_5m_spacing.h5'
# d = DatasetConsumer(DATASET)
# d.print_info()
# paths = d.generate_straight_paths(200)

# dataset = d.paths_to_dataset_mag_only(paths)
# print(dataset.shape)