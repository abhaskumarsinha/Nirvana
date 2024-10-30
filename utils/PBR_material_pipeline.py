import numpy as np


from Nirvana.utils.barycentric_function import *
from Nirvana.utils.draw_canvas import *
from Nirvana.utils.PBR import *

from Nirvana.config.config import *




def PBR_pipeline_texture(canvas,
                         face,
                         texture,
                         oa,
                         normal_tex,
                         roughness,
                         gloss,
                         uv,
                         normal,
                         view_direction,
                         light_direction,
                         H,
                         fresnel_value,
                         pixel_density):
    # write code here

    # Convert all textures to float before dividing by 255
    texture = texture.astype(np.float32) / 255.0
    oa = oa.astype(np.float32) / 255.0
    normal_tex = normal_tex.astype(np.float32) / 255.0
    roughness = roughness.astype(np.float32) / 255.0
    gloss = gloss.astype(np.float32) / 255.0


    min_x, min_y = np.min(face, axis=0)
    max_x, max_y = np.max(face, axis=0)

    num_samples_x = int(pixel_density * (max_x - min_x + 1))
    num_samples_y = int(pixel_density * (max_y - min_y + 1))

    # Create a grid of points within the bounding box using the specified pixel density
    x_range = np.linspace(min_x, max_x, num_samples_x)
    y_range = np.linspace(min_y, max_y, num_samples_y)

    for x in x_range:
        for y in y_range:
            p = np.array([x, y])
            u, v, w = barycentric_coords(p, face[0], face[1], face[2])

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
                diffuse_pix = texture[tex_y, tex_x]
                oa_pix = oa[tex_y, tex_x]
                normal_pix = normal_tex[tex_y, tex_x]
                roughness_pix = roughness[tex_y, tex_x]
                gloss_pix = gloss[tex_y, tex_x]

                PBR_shader_pix = cook_torrance_brdf(normal, 
                                                    view_direction, 
                                                    light_direction, 
                                                    H, 
                                                    0.2,          #roughness_pix, 
                                                    0.2,          #gloss_pix, 
                                                    fresnel_value)
                #print('PBR Shader: ', PBR_shader_pix * 4)
                
                normal_pix = normal_pix * 2 - 1
                normal_pix /= np.linalg.norm(normal_pix, axis = -1)
                normal_pix = np.clip(np.sum(normal_pix * light_direction, axis=-1), 0, 1)
                normal_pix = 1 - normal_pix
                #print('normal with light direction: ', normal_pix)

                diffuse_pix = diffuse_pix * PBR_shader_pix * normal_pix * oa_pix *10
                #print('diffuse after OA PBR and normal: ', diffuse_pix)

                final_color = np.clip(diffuse_pix, 0, 1)

                plot_pixel(canvas = canvas, x = x, y = y, color = final_color, x_scene = (-10, 10), y_scene = (-10, 10))

    
