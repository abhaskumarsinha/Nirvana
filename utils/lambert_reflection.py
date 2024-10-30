import numpy as np
import matplotlib.pyplot as plt
from Nirvana.utils.barycentric_function import *

def lambert_pipeline(face, uv, texture, light_value, canvas, render_resolution, face_range):
    texture = texture.astype(np.float32) / 255.0
    
    # Define the bounding box within the face range
    min_face_x = np.min(face[:, 0])
    max_face_x = np.max(face[:, 0])
    min_face_y = np.min(face[:, 1])
    max_face_y = np.max(face[:, 1])

    # Set the fixed range for rendering the 3D scene as provided
    range_x, range_y = 20, 20  # This range is essential for the rendering area as per your requirement

    # Step sizes within the rendering resolution space
    step_size_x = range_x / render_resolution[0]
    step_size_y = range_y / render_resolution[1]

    # Start at the minimum x and y values of the face within the fixed range
    current_x = min_face_x
    while current_x < max_face_x:
        current_y = min_face_y
        while current_y < max_face_y:
            print('Running while loop!', current_y)

            # Calculate the 2D coordinates in the canvas space
            canvas_x = int(((current_x - (min_face_x - range_x / 2)) / range_x) * render_resolution[0])
            canvas_y = int(((current_y - (min_face_y - range_y / 2)) / range_y) * render_resolution[1])

            # Compute the barycentric coordinates for the current point
            p = np.array([current_x, current_y])
            u, v, w = barycentric_coords(p, face[0], face[1], face[2])
            
            # Check if the point is within the triangle
            if u >= 0 and v >= 0 and w >= 0:
                tx, ty = u * uv[0] + v * uv[1] + w * uv[2]
                tx = int(tx * (texture.shape[1] - 1))
                ty = int(ty * (texture.shape[0] - 1))

                # Ensure the coordinates are within the canvas bounds
                if 0 <= canvas_x < render_resolution[0] and 0 <= canvas_y < render_resolution[1]:
                    canvas[canvas_x, canvas_y] = texture[ty, tx] * light_value
                    print('Changing canvas colors')

            # Move to the next y-step
            current_y += step_size_y
        # Move to the next x-step
        current_x += step_size_x
