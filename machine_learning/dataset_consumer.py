import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
from typing import Callable
from cprint import *

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
        self.ray_aoas = None

        with h5py.File(dataset_path, 'r') as file:
            self.attributes = self.__clean_attributes(file.attrs)
            self.csi_mags = file['csis_mag'][:]
            self.csi_phases = file['csis_phase'][:]
            self.rx_positions = file['positions'][:]
            self.ray_aoas = file['ray_aoas'][:]

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


    def __closest_real_index(self, x, y):
        # Find the closest point in the rx_positions array
        index = np.argmin(np.abs(self.rx_positions[0, :] - x) + np.abs(self.rx_positions[1, :] - y))
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
    
    # Descale data with passed function
    def descale(self, scaler: Callable[[np.array], np.array], data: np.array) -> np.array:
        return scaler(data)
    
    # Unwrap data with passed function
    def unwrap(self, data: np.array, axis:int=0) -> np.array:
        return np.unwrap(data, axis=axis)
    
    # Convert angles to trigonometric form
    def to_trig(self, data: np.array) -> np.array:
        return np.array([np.cos(data * 2 * np.pi / 360), np.sin(data * 2 * np.pi / 360)])

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
    
    def paths_to_dataset_rays_aoas(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, num_rays)
        """
        # Use the indices to grab the positions for each point
        cprint.info(f'self.ray_aoas.shape {self.ray_aoas.T.shape}')
        return self.ray_aoas.T[path_indices, 0, :], self.ray_aoas.T[path_indices, 1, :]
    
    def paths_to_dataset_rays_aoas_trig(self, path_indices, pad = 0):
        """
        Generate a torch dataset from the given path indices
        """
        # Use the indices to grab the positions for each point
        cprint.info(f'self.ray_aoas.shape {self.ray_aoas.T.shape}')
        azimuth_cos = np.cos(self.ray_aoas.T[:, 0, :] * 2* np.pi / 360)
        azimuth_cos = np.pad(azimuth_cos, ((0,0), (0,pad)))
        azimuth_sin = np.sin(self.ray_aoas.T[:, 0, :] * 2* np.pi / 360)
        azimuth_sin = np.pad(azimuth_sin, ((0,0), (0,pad)))
        elevation_cos = np.cos(self.ray_aoas.T[:, 1, :] * 2* np.pi / 360)
        elevation_cos = np.pad(elevation_cos, ((0,0), (0,pad)))
        elevation_sin = np.sin(self.ray_aoas.T[:, 1, :] * 2* np.pi / 360)
        elevation_sin = np.pad(elevation_sin, ((0,0), (0,pad)))
        cprint.info(f'azimuth_cos.shape {azimuth_cos.shape}')
        cprint.info(f'azimuth_sin.shape {azimuth_sin.shape}')
        cprint.info(f'elevation_cos.shape {elevation_cos.shape}')
        cprint.info(f'elevation_sin.shape {elevation_sin.shape}')
        return np.array([azimuth_cos[path_indices, :], azimuth_sin[path_indices, :], elevation_cos[path_indices, :], elevation_sin[path_indices, :]])

    def paths_to_dataset_positions(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 2)
        """
        # Use the indices to grab the positions for each point
        positions = self.rx_positions[:, path_indices]
        positions = np.swapaxes(positions, 0, 1)
        return positions
    
    def paths_to_dataset_interleaved(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 256)
        """
        # Get the magnitude and phase data
        csi_mags = self.paths_to_dataset_mag_only(path_indices)
        csi_phases = self.paths_to_dataset_phase_only(path_indices)


        # Create a new array to hold the interleaved data
        num_paths, path_length_n, _ = csi_mags.shape
        interleaved = np.empty((num_paths, path_length_n, 256), dtype=csi_mags.dtype)

        # Fill the new array with alternating slices from the two original arrays
        interleaved[..., ::2] = csi_mags
        interleaved[..., 1::2] = csi_phases

        return interleaved
    
    # def paths_to_dataset_interleaved_padded(self, path_indices):
    #     """
    #     Generate a numpy dataset from the given path indices
    #     Shape: (num_paths, path_length_n, 384)
    #     """
    #     # Get the magnitude, phase, and rays_aoas_trig data
    #     csi_mags = self.paths_to_dataset_mag_only(path_indices)
    #     csi_phases = self.paths_to_dataset_phase_only(path_indices)
    #     rays_aoas_trig = self.paths_to_dataset_rays_aoas_trig(path_indices)
    #     cprint.warn(f'csi_mags.shape {csi_mags.shape}')
    #     cprint.warn(f'rays_aoas_trig.shape {rays_aoas_trig.shape}')
    #     # Find the maximum length
    #     max_length = max(csi_mags.shape[-1], csi_phases.shape[-1], rays_aoas_trig.shape[-1])

    #     # Pad the arrays to match the maximum length
    #     csi_mags = np.pad(csi_mags, ((0,0), (0,0), (0, max_length - csi_mags.shape[-1])))
    #     csi_phases = np.pad(csi_phases, ((0,0), (0,0), (0, max_length - csi_phases.shape[-1])))
    #     rays_aoas_trig = np.pad(rays_aoas_trig, ((0,0), (0,0), (0, max_length - rays_aoas_trig.shape[-1])))

    #     # Interleave the arrays
    #     interleaved = np.empty((csi_mags.shape[0], csi_mags.shape[1], 3 * max_length))
    #     interleaved[..., ::3] = csi_mags
    #     interleaved[..., 1::3] = csi_phases
    #     interleaved[..., 2::3] = rays_aoas_trig

    #     return interleaved

    def paths_to_dataset_interleaved_w_rays(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 256)
        """
        # Get the magnitude and phase data
        csi_mags = self.paths_to_dataset_mag_only(path_indices)
        csi_phases = self.paths_to_dataset_phase_only(path_indices)


        # Create a new array to hold the interleaved data
        num_paths, path_length_n, _ = csi_mags.shape
        interleaved = np.empty((num_paths, path_length_n, 256), dtype=csi_mags.dtype)

        # Fill the new array with alternating slices from the two original arrays
        interleaved[..., ::2] = csi_mags
        interleaved[..., 1::2] = csi_phases
        
        interleaved_all = np.empty((num_paths, path_length_n, 656), dtype=csi_mags.dtype)
        interleaved_all[..., :256] = interleaved
        rays_aoas_trig = self.paths_to_dataset_rays_aoas_trig(path_indices)
        interleaved_all[..., 256:356] = rays_aoas_trig[0,:,:,:]
        interleaved_all[..., 356:456] = rays_aoas_trig[1,:,:,:]
        interleaved_all[..., 456:556] = rays_aoas_trig[2,:,:,:]
        interleaved_all[..., 556:656] = rays_aoas_trig[3,:,:,:]
        cprint.warn(f'interleaved_all.shape {interleaved_all.shape}')
        return interleaved_all
        
    def add_terminals_to_paths(self, path_indices, distance_from_end=1):
        """
        For each path, add a left and right terminal.

        These are just added as the last two points in the path
        (left terminal is second to last, right terminal is last)
        """
        num_paths, path_length_n = paths.shape
        paths_with_terminals = np.empty((num_paths, path_length_n + 2), dtype=paths.dtype)
        for i in range(num_paths):
            # Add the base path
            paths_with_terminals[i, :-2] = paths[i]

            # Find the direction at the end of the path
            last_point = self.rx_positions[:, paths[i, -1]]
            second_to_last_point = self.rx_positions[:, paths[i, -4]]
            direction = last_point - second_to_last_point
            direction = direction / np.linalg.norm(direction)

            left_direction = np.array([-direction[1], direction[0], 0])
            right_direction = np.array([direction[1], -direction[0], 0])

            # Find the left and right terminals
            left_terminal = last_point + (left_direction * distance_from_end)
            right_terminal = last_point + (right_direction * distance_from_end)

            # Find the closest point in the rx_positions array
            left_terminal_index = self.__closest_real_index(left_terminal[0], left_terminal[1])
            right_terminal_index = self.__closest_real_index(right_terminal[0], right_terminal[1])

            # Add the terminals to the path
            paths_with_terminals[i, -2] = left_terminal_index
            paths_with_terminals[i, -1] = right_terminal_index

        return paths_with_terminals
    
# DATASET = 'older_ver/dataset_0_5m_spacing.h5'
# d = DatasetConsumer(DATASET)
# d.print_info()

# paths = d.generate_straight_paths(200)
# paths = d.add_terminals_to_paths(paths, distance_from_end=2)
# pos = d.paths_to_dataset_positions(paths)

# # Plot the first 50 paths
# for i in range(50):
#     plt.plot(pos[i, 0, :], pos[i, 1, :])
# plt.show()