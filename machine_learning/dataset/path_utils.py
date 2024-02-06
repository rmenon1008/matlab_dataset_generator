import numpy as np


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


def curve_in_range(delta, s, theta, x, y, range_of_points, L, dt):
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
        dy = np.sin(theta) * s * dt
        dtheta = (s / L) * np.tan(delta) * dt
        xnew = x + dx
        ynew = y + dy
        thetanew = theta + dtheta
        thetanew = np.mod(thetanew, 2 * np.pi)  # Wrap theta at pi
        xvec.append(xnew)
        yvec.append(ynew)
        points.append((xnew, ynew))
        thetavec.append(thetanew)
        x = xnew
        y = ynew
        theta = thetanew
    # print(f"Generated Points: {points}")
    # print(f"Generated Angles: {thetavec}")

    return [xvec, yvec, points, thetavec]
