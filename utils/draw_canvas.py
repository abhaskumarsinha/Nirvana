import numpy as np

def plot_pixel_interpolated(canvas, x, y, color, x_scene=(-10, 10), y_scene=(-10, 10)):
    """
    Plots a single point in a given canvas array using bilinear interpolation.

    Parameters:
        canvas (numpy.ndarray): The canvas array of shape (x_size, y_size, 3).
        x (float): The x-coordinate of the point to plot.
        y (float): The y-coordinate of the point to plot.
        color (tuple): RGB color as a tuple of (R, G, B), each value 0-255.
        x_scene (tuple): The range of x in the scene, default (-10, 10).
        y_scene (tuple): The range of y in the scene, default (-10, 10).
    """
    x_size, y_size, _ = canvas.shape
    
    # Scaling factors to map scene coordinates to canvas indices
    x_scale = x_size / (x_scene[1] - x_scene[0])
    y_scale = y_size / (y_scene[1] - y_scene[0])
    
    # Calculate float indices on canvas
    i_float = (x - x_scene[0]) * x_scale
    j_float = (y - y_scene[0]) * y_scale
    
    # Calculate the integer parts and the fractional parts
    i0, j0 = int(np.floor(i_float)), int(np.floor(j_float))
    di, dj = i_float - i0, j_float - j0
    
    # Ensure we are within bounds
    if 0 <= i0 < x_size - 1 and 0 <= j0 < y_size - 1:
        # Get weights for interpolation
        w00 = (1 - di) * (1 - dj)
        w01 = (1 - di) * dj
        w10 = di * (1 - dj)
        w11 = di * dj
        
        # Blend the color into the four nearest pixels
        canvas[i0, j0] = (canvas[i0, j0] * (1 - w00) + np.array(color) * w00).astype(np.uint8)
        canvas[i0, j0 + 1] = (canvas[i0, j0 + 1] * (1 - w01) + np.array(color) * w01).astype(np.uint8)
        canvas[i0 + 1, j0] = (canvas[i0 + 1, j0] * (1 - w10) + np.array(color) * w10).astype(np.uint8)
        canvas[i0 + 1, j0 + 1] = (canvas[i0 + 1, j0 + 1] * (1 - w11) + np.array(color) * w11).astype(np.uint8)
