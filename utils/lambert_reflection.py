import numpy as np
import matplotlib.pyplot as plt

def lambert_pipeline(vertices,
                     uv,
                     texture,
                     light_value,
                     ax,
                     pixel_density=10,
                     resolution=(100, 100),  # 100x100 pixel resolution for imshow
                     xy_range=10):
    """
    Render a textured face with lighting onto the matplotlib axis `ax` using imshow.
    vertices: 3x2 array of triangle vertices (in screen space).
    uv: 3x2 array of UV coordinates for the vertices.
    texture: The texture image to project onto the triangle.
    light_value: 1x3 array of RGB light intensity values for the triangle.
    ax: The matplotlib axis to render on.
    pixel_density: The resolution of the pixels per unit area.
    resolution: Tuple indicating the pixel resolution of the output image.
    xy_range: The coordinate range to map to pixel space.
    """
    # Initialize a blank canvas with the desired resolution
    canvas = np.ones((resolution[0], resolution[1], 3))
    texture = texture / 255  # Normalize texture values to [0, 1]

    # Calculate scaling from coordinate space to pixel space
    scale_x = resolution[1] / (2 * xy_range)
    scale_y = resolution[0] / (2 * xy_range)

    # Get bounding box of triangle and set up pixel grid
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    x_range = np.linspace(min_x, max_x, int((max_x - min_x) * pixel_density))
    y_range = np.linspace(min_y, max_y, int((max_y - min_y) * pixel_density))

    for x in x_range:
        for y in y_range:
            p = np.array([x, y])
            u, v, w = barycentric_coords(p, vertices[0], vertices[1], vertices[2])

            # Check if point is inside triangle
            if u >= 0 and v >= 0 and w >= 0:

                # Interpolate UV coordinates and get texture color
                uv_p = u * uv[0] + v * uv[1] + w * uv[2]
                tex_x = int(uv_p[0] * (texture.shape[1] - 1))
                tex_y = int(uv_p[1] * (texture.shape[0] - 1))
                tex_x = np.clip(tex_x, 0, texture.shape[1] - 1)
                tex_y = np.clip(tex_y, 0, texture.shape[0] - 1)
                tex_color = texture[tex_y, tex_x]

                # Apply lighting
                final_color = np.clip(tex_color * light_value, 0, 1)

                # Map (x, y) to canvas indices
                pixel_x = int((x + xy_range) * scale_x)
                pixel_y = int((y + xy_range) * scale_y)
                
                if 0 <= pixel_x < resolution[1] and 0 <= pixel_y < resolution[0]:
                    canvas[pixel_y, pixel_x] = final_color

    # Display the canvas using imshow
    ax.imshow(canvas, extent=(-xy_range, xy_range, -xy_range, xy_range))
