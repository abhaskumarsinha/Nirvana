import numpy as np
import matplotlib.pyplot as plt

from Nirvana.utils.barycentric_function import *

def lambert_pipeline(face, 
                     uv, 
                     texture, 
                     light_value, 
                     canvas, 
                     render_resolution, 
                     face_range):

    min_face_x = np.min(face[:, 0])
    max_face_x = np.max(face[:, 0])
    min_face_y = np.min(face[:, 1])
    max_face_y = np.max(face[:, 1])

    range_x = 20
    range_y = 20

    step_size_x = range_x / render_resolution[0]
    step_size_y = range_x / render_resolution[1]

    while min_face_x < max_face_x:
        while min_face_y < max_face_y:

            x = int((min_face_x / range_x + 0.5) * render_resolution[0])
            y = int((min_face_y / range_y + 0.5) * render_resolution[1])
            
            p = np.array([x, y])

            u, v, w = barycentric_coords(p, face[0], face[1], face[2])
            print('found barycentric coords: ', u, v, w)
            if u >= 0 and v >= 0 and w >= 0:
                tx, ty = u * uv[0] + v * uv[1] + w * uv[2]
                tx = int(tx * (image.shape[1] - 1))
                ty = int(ty * (image.shape[0] - 1))

                canvas[x, y] = image[ty, tx]
                print('Printing color: ', image[ty, tx], ' at: ', x, y)
                
            min_face_y += step_size_y
        min_face_x += step_size_x

