import numpy as np
import matplotlib.pyplot as plt
from Nirvana.utils.barycentric_function import *

def lambert_pipeline(face, uv, texture, light_value, canvas, render_resolution, face_range):
    texture = texture.astype(np.float32) / 255.0
    
    # Determine bounding box dimensions
    min_face_x, min_face_y = np.min(face[:, 0]), np.min(face[:, 1])
    max_face_x, max_face_y = np.max(face[:, 0]), np.max(face[:, 1])

    # Use the actual bounding box dimensions instead of a fixed range
    range_x, range_y = max_face_x - min_face_x, max_face_y - min_face_y

    # Step sizes based on render resolution and bounding box size
    step_size_x = range_x / render_resolution[0]
    step_size_y = range_y / render_resolution[1]

    # Iterate over the face's bounding box area
    x = min_face_x
    while x < max_face_x:
        y = min_face_y
        while y < max_face_y:
            print('Running while loop!', y)

            # Calculate canvas coordinates
            canvas_x = int(((x - min_face_x) / range_x) * render_resolution[0])
            canvas_y = int(((y - min_face_y) / range_y) * render_resolution[1])

            # Calculate barycentric coordinates
            p = np.array([x, y])
            u, v, w = barycentric_coords(p, face[0], face[1], face[2])
            
            # Only draw if the point is inside the triangle
            if u >= 0 and v >= 0 and w >= 0:
                tx, ty = u * uv[0] + v * uv[1] + w * uv[2]
                tx = int(tx * (texture.shape[1] - 1))
                ty = int(ty * (texture.shape[0] - 1))

                # Ensure the coordinates are within the canvas bounds
                if 0 <= canvas_x < render_resolution[0] and 0 <= canvas_y < render_resolution[1]:
                    canvas[canvas_x, canvas_y] = texture[ty, tx] * light_value
                    print('Changing canvas colors')
                
            y += step_size_y
        x += step_size_x
