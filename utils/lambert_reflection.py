import numpy as np
import matplotlib.pyplot as plt
from Nirvana.utils.barycentric_function import *

def lambert_pipeline(vertices,
                     uv,
                     texture,
                     light_value,
                     ax,
                     pixel_density=10,
                     resolution=(420, 420)):
    """
    Render a textured face with lighting onto a 2D canvas and display with imshow.
    vertices: 3x2 array of triangle vertices (in screen space).
    uv: 3x2 array of UV coordinates for the vertices.
    texture: The texture image to project onto the triangle.
    light_value: 1x3 array of RGB light intensity values for the triangle.
    ax: The matplotlib axis to render on.
    pixel_density: The resolution of the pixels per unit area. (Default = 10)
    resolution: The (height, width) resolution of the output canvas.
    """
    texture = texture / 255
    height, width = resolution

    # Initialize a blank RGB canvas
    canvas = np.zeros((height, width, 3))

    # Get the bounding box of the triangle
    min_x, min_y = np.min(vertices, axis=0).astype(int)
    max_x, max_y = np.max(vertices, axis=0).astype(int)

    # Clip bounding box to stay within canvas resolution
    min_x, max_x = np.clip([min_x, max_x], 0, width - 1)
    min_y, max_y = np.clip([min_y, max_y], 0, height - 1)

    # Generate coordinates for each pixel within bounding box
    for i in range(min_y, max_y + 1):
        for j in range(min_x, max_x + 1):
            p = np.array([j, i])
            u, v, w = barycentric_coords(p, vertices[0], vertices[1], vertices[2])

            # If point is within the triangle
            if u >= 0 and v >= 0 and w >= 0:
                # Interpolate UV coordinates
                uv_p = u * uv[0] + v * uv[1] + w * uv[2]

                # Map UV to texture space
                tex_x = int(uv_p[0] * (texture.shape[1] - 1))
                tex_y = int(uv_p[1] * (texture.shape[0] - 1))
                tex_x = np.clip(tex_x, 0, texture.shape[1] - 1)
                tex_y = np.clip(tex_y, 0, texture.shape[0] - 1)

                # Get the color from texture and apply lighting
                tex_color = texture[tex_y, tex_x]
                final_color = np.clip(tex_color * light_value, 0, 1)

                # Assign color to canvas pixel
                canvas[i, j] = final_color

    # Display the filled canvas
    ax.imshow(canvas, origin='lower')
