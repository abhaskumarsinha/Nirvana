import numpy as np


from Nirvana.utils.barycentric_function import *
from Nirvana.utils.draw_canvas import *
from Nirvana.utils.PBR import *

from Nirvana.config.config import *


def PBR_material_pipeline(canvas,
                         vertices,
                         obj,
                         fresnel_value,
                         light_configs,
                         ax,
                         pixel_density = 10):
    """
    Render a textured face with lighting onto the matplotlib axis `ax`.
    vertices: 3x2 array of triangle vertices (in screen space).
    uv: 3x2 array of UV coordinates for the vertices.
    texture: The texture image to project onto the triangle.
    light_value: 1x3 array of RGB light intensity values for the triangle.
    ax: The matplotlib axis to render on.
    pixel_density: The resolution of the pixels per unit area. (Default = 10)
    """
                           
    texture = obj['material'].get_diffuse_texture().astype(float) / 255
    normal = obj['material'].get_normal_texture().astype(float) / 255
    ao = obj['material'].get_ao_texture().astype(float) / 255
    roughness = obj['material'].get_roughness_texture().astype(float) / 255
    metallic = obj['material'].get_metallic_texture().astype(float) / 255
    fresnel = fresnel_value

    L, V, N, H = light_configs   
    print('half angle: ', H)
    print('tangents: ', N)
    print('face_color: ', cook_torrance_brdf(N, V, L, H, 0.5, 0, fresnel_value))

    normal_rescaled = (normal * 2) - 1
    dot_product = np.einsum('ijk,k->ij', normal_rescaled, L)
    shadow_intensity = np.clip(dot_product, 0, 1)


    uv = obj['uv_map']
                           
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
                R = roughness[tex_y, tex_x]
                G = metallic[tex_y, tex_x]
                shadow = shadow_intensity[tex_y, tex_x]
                AO_shadow = ao[tex_y, tex_x]
                final_color = ggx_distribution(np.dot(N, H.T), 0.6)
                #final_color = cook_torrance_brdf(N, V, L, H, R/2, G/2, fresnel_value)
                #final_color *= tex_color
                #final_color *= shadow
                #final_color *= AO_shadow
                #final_color *= 4
                
                # Ensure the final color stays within valid bounds [0, 1]
                final_color = np.clip(final_color, 0, 1)

                # Plot the pixel
                plot_pixel(canvas = canvas, x = y, y = x, color = final_color, x_scene = (-10, 10), y_scene = (-10, 10))
