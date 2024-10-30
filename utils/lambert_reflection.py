import numpy as np
from Nirvana.utils.barycentric_function import *
from Nirvana.utils.draw_canvas import *

def lambert_pipeline(    canvas,
                         vertices,
                         uv,
                         texture,
                         light_value,
                         ax,
                         pixel_density=10):
    """
    Render a textured face with lighting onto the matplotlib axis `ax`.
    vertices: 3x2 array of triangle vertices (in screen space).
    uv: 3x2 array of UV coordinates for the vertices.
    texture: The texture image to project onto the triangle.
    light_value: 1x3 array of RGB light intensity values for the triangle.
    ax: The matplotlib axis to render on.
    pixel_density: The resolution of the pixels per unit area. (Default = 10)
    """
    texture = texture / 255
    # Get the bounding box of the triangle
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)

    # Ensure that we use integers for the number of samples
    num_samples_x = int(pixel_density * (max_x - min_x + 1))
    num_samples_y = int(pixel_density * (max_y - min_y + 1))

    # Create a grid of points within the bounding box using the specified pixel density
    x_range = np.linspace(min_x, max_x, num_samples_x)
    y_range = np.linspace(min_y, max_y, num_samples_y)

    for x in x_range:
        for y in y_range:
            p = np.array([x, y])
            u, v, w = barycentric_coords(p, vertices[0], vertices[1], vertices[2])

            # If the point is inside the triangle (u, v, w >= 0 and u + v + w == 1)
            if u >= 0 and v >= 0 and w >= 0:

                # Interpolate the UV coordinates
                uv_p = u * uv[0] + v * uv[1] + w * uv[2]

                # Map the UV coordinates to texture space
                tex_x = int(uv_p[0] * (texture.shape[1] - 1))
                tex_y = int(uv_p[1] * (texture.shape[0] - 1))

                # Ensure the texture coordinates are within bounds
                tex_x = np.clip(tex_x, 0, texture.shape[1] - 1)
                tex_y = np.clip(tex_y, 0, texture.shape[0] - 1)

                # Sample the color from the texture
                tex_color = texture[tex_y, tex_x]

                # Apply lighting by modulating the texture color with the light value
                final_color = tex_color * light_value

                # Ensure the final color stays within valid bounds [0, 1]
                final_color = np.clip(final_color, 0, 1)

                # Plot the pixel
                plot_pixel(canvas = canvas, x = x, y = y, x_scene = (-10, 10), y_scene = (-10, 10))
