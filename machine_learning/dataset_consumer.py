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

def curve_in_range(delta, s, theta,x, y, range_of_points, L,dt):
    """
    Generate a curved line between two points using a method similar to car kinematics
    Inputs: 
        delta - steering angle (radians)
        s     - speed
        theta - car heading (radians)
        x     - x coordinates of start and end points
        y     - y coordinates of start and end points
        range - the number of points we want to have
        L     - Length of car (omitting this?)
        dt    - time increment   
    Returns: 2 lists, one with the (x, y) points to create the line, 
        and another with the heading angles 
    """
    xvec = []
    yvec = []
    points = []
    thetavec = []
    for i in range(len(range_of_points)):
        dx = np.cos(theta) * s * dt
        dy = np.sin(theta)* s *dt
        dtheta = (s/L) * np.tan(delta) * dt
        xnew = x + dx
        ynew = y + dy
        thetanew = theta + dtheta
        thetanew = np.mod(thetanew,2*np.pi) # Wrap theta at pi
        xvec.append(xnew)
        yvec.append(ynew)
        points.append((xnew, ynew))
        thetavec.append(thetanew)
        x = xnew
        y= ynew
        theta = thetanew
    # print(f"Generated Points: {points}")
    # print(f"Generated Angles: {thetavec}")

    return([xvec, yvec, points, thetavec]) 

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
    
    def __single_straight_path(self, start_point, end_point):
        # Generate a straight path between two points
        x0, y0 = self.__real_to_grid(start_point[0], start_point[1])
        x1, y1 = self.__real_to_grid(end_point[0], end_point[1])
        points = breesenham(x0, y0, x1, y1)
        return np.array([self.__grid_to_real_index(x, y) for x, y in points])
    
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
                path = self.__single_straight_path(self.rx_positions[:, point1], self.rx_positions[:, point2])

                # Make sure the path is at least path_length_n long
                if len(path) <= path_length_n:
                    continue

                path_indices[i] = path[:path_length_n]

                break
        return path_indices
    

    def generate_curved_paths(self, num_paths, path_length_n=20):
        """
        Generate curved paths in the rx_positions array. (currently not working as expected)

        inputs:
            num_paths: Number of paths to generate
            path_length_n: Length of each path in number of points
        output: An array with shape: (num_paths, path_length_n)
        """
        deg2rad = np.pi / 180.0
        rad2deg = 180.0 / np.pi
        print(f'Generating {num_paths} paths of length {path_length_n}')
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
                dt = 1 # this value can be changed, currently the time step is 1

                # Generate a curved line between from random point and set angle
                x, y = self.__real_to_grid(self.rx_positions[0, point1], self.rx_positions[1, point1])
                range_of_points = np.arange(0,path_length_n,dt)
                [x_coor, y_coor, points, thetas] = curve_in_range(delta, 1, theta, x, y, range_of_points, 0.2, dt)
                
                # Check if all the points are within the grid bounds
                ep = 10
                if np.any(np.array(x_coor) < self.grid_size[0] + ep) or np.any(np.array(x_coor) > self.grid_size[1] - ep):
                    continue
                if np.any(np.array(y_coor) < self.grid_size[2] + ep) or np.any(np.array(y_coor) > self.grid_size[3] - ep):
                    continue

                # Make sure the path is at least path_length_n long
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
    
    def paths_to_relative_positions(self, path_indices):
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
        return positions[:, 0:2, :]
    
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
    

    def paths_to_dataset_interleaved_w_relative_positions(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 258)
        """
        # Get the magnitude and phase data
        csi_mags = self.paths_to_dataset_mag_only(path_indices)
        csi_phases = self.paths_to_dataset_phase_only(path_indices)
        positions = self.paths_to_relative_positions(path_indices)
        positions = np.swapaxes(positions, 1, 2)

        # Create a new array to hold the interleaved data
        num_paths, path_length_n, _ = csi_mags.shape
        interleaved = np.empty((num_paths, path_length_n, 256), dtype=csi_mags.dtype)

        # Fill the new array with alternating slices from the two original arrays
        interleaved[..., ::2] = csi_mags
        interleaved[..., 1::2] = csi_phases

        # Add the positions to the end of the array
        concatenated = np.concatenate((interleaved, positions), axis=2)
        return concatenated
    
    
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
        
    def create_left_center_right_paths(self, path_indices, terminal_length=1):
        """
        Each path is replaced with 3 paths. Each has the same starting number of points as the original path.
        Then, each path is extended by a terminal_length number of points in 3 different directions.
        """
        def rotate_vec(vec, angle):
            """
            Rotate a vector by the given angle
            """
            flat = np.array([vec[0], vec[1]])
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            angled = np.matmul(rotation_matrix, flat)
            return np.array([angled[0], angled[1], vec[2]])

        num_paths, path_length_n = path_indices.shape
        left_paths = np.zeros((num_paths, path_length_n + terminal_length), dtype=np.int32)
        center_paths = np.zeros((num_paths, path_length_n + terminal_length), dtype=np.int32)
        right_paths = np.zeros((num_paths, path_length_n + terminal_length), dtype=np.int32)
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

            # Choose directions for the terminals
            left_direction = rotate_vec(direction, np.pi / 4)
            center_direction = direction
            right_direction = rotate_vec(direction, -np.pi / 4)

            # Find the end points of the terminals
            left_end = last_point + left_direction * terminal_length * self.grid_spacing * 2
            center_end = last_point + center_direction * terminal_length * self.grid_spacing * 2
            right_end = last_point + right_direction * terminal_length * self.grid_spacing * 2

            # Find the indices of the end points
            left_index = self.__closest_real_index(left_end[0], left_end[1])
            center_index = self.__closest_real_index(center_end[0], center_end[1])
            right_index = self.__closest_real_index(right_end[0], right_end[1])

            # Create straight paths to the end points
            left_path = self.__single_straight_path(last_point, self.rx_positions[:, left_index])
            center_path = self.__single_straight_path(last_point, self.rx_positions[:, center_index])
            right_path = self.__single_straight_path(last_point, self.rx_positions[:, right_index])

            # Repeat the last point if terminal is too short
            def pad_path(path):
                if len(path) < terminal_length:
                    path = np.repeat(path[-1:], terminal_length, axis=0)
                if len(path) > terminal_length:
                    path = path[:terminal_length]
                return path
            
            left_path = pad_path(left_path)
            center_path = pad_path(center_path)
            right_path = pad_path(right_path)

            # Add the terminals to the path
            left_paths[i, path_length_n:] = left_path
            center_paths[i, path_length_n:] = center_path
            right_paths[i, path_length_n:] = right_path

        return (left_paths, center_paths, right_paths)
    
# DATASET = 'older_ver/dataset_0_5m_spacing.h5'
# d = DatasetConsumer(DATASET)
# d.print_info()

# # Start with curved paths
# paths = d.generate_curved_paths(200, path_length_n=20)

# # Create left, center, and right paths
# left_paths, center_paths, right_paths = d.create_left_center_right_paths(paths, terminal_length=10)

# # Create datasets from the paths
# left_dataset = d.paths_to_dataset_interleaved_w_relative_positions(left_paths)
# center_dataset = d.paths_to_dataset_interleaved_w_relative_positions(center_paths)
# right_dataset = d.paths_to_dataset_interleaved_w_relative_positions(right_paths)

# # Plot the first 50 paths' relative positions
# for i in range(50):
#     plt.plot(left_dataset[i, :, -1], left_dataset[i, :, -2])
#     plt.plot(right_dataset[i, :, -1], right_dataset[i, :, -2])
#     plt.plot(center_dataset[i, :, -1], center_dataset[i, :, -2])
# plt.show()