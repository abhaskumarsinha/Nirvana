import numpy as np

def plot_pixel(canvas, x, y, color, x_scene=(-10, 10), y_scene=(-10, 10), 
               upside_down=False, left_right=True):
    """
    Plots a single point in a given canvas array of specified resolution,
    with options to flip the y-axis (upside-down) and/or the x-axis (left-right).

    Parameters:
        canvas (numpy.ndarray): The canvas array of shape (x_size, y_size, 3).
        x (float): The x-coordinate of the point to plot.
        y (float): The y-coordinate of the point to plot.
        color (tuple): RGB color as a tuple of (R, G, B), each value 0-255.
        x_scene (tuple): The range of x in the scene, default (-10, 10).
        y_scene (tuple): The range of y in the scene, default (-10, 10).
        upside_down (bool): If True, flip the y-axis for upright orientation. Default is True.
        left_right (bool): If True, flip the x-axis for mirror image. Default is False.
    """
    x_size, y_size, _ = canvas.shape
    
    # Scaling factors to map scene coordinates to canvas indices
    x_scale = x_size / (x_scene[1] - x_scene[0])
    y_scale = y_size / (y_scene[1] - y_scene[0])
    
    # Map float x, y coordinates to integer canvas indices
    i = int(round((x - x_scene[0]) * x_scale))
    j = int(round((y - y_scene[0]) * y_scale))
    
    # Apply left-right flip if specified
    if left_right:
        i = x_size - 1 - i
    
    # Apply upside-down flip if specified
    if upside_down:
        j = y_size - 1 - j
    
    # Ensure i, j are within bounds of the canvas
    if 0 <= i < x_size and 0 <= j < y_size:
        canvas[i, j] += color
