import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from tqdm import tqdm

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileShader, compileProgram

from Nirvana.camera.camera import *
from Nirvana.lights.light import *
from Nirvana.objects.base import *
from Nirvana.utils import *
from Nirvana.lights import *

import cv2


# Move the function below to a different file
# GPU FUNCTIONS! DON'T TOUCH!
def draw_wireframe(vertices, faces):
    """
    Draw the wireframe of a 3D shape.
    vertices: A numpy array of vertex positions.
    faces: A numpy array of indices into the vertices array that define the faces.
    """
    glBegin(GL_LINES)  # We will draw lines for the wireframe
    for face in faces:
        for i in range(len(face)):
            # Draw each edge by connecting vertices
            start_vertex = vertices[face[i]]
            end_vertex = vertices[face[(i + 1) % len(face)]]  # Wrap around to first vertex
            glVertex3fv(start_vertex)
            glVertex3fv(end_vertex)
    glEnd()

def draw_solid_faces(vertices, faces, normals, lights):
    """
    Draw solid faces with shading based on the lights in the scene.
    
    vertices: A numpy array of vertex positions.
    faces: A numpy array of indices into the vertices array that define the faces.
    normals: A numpy array of face normals (not normalized).
    lights: A list of light objects, where each object has:
        - .orientation (numpy array of shape (3,)) representing the light direction
        - .intensity (float scalar representing the intensity of the light)
        - .color (numpy array of shape (3,) representing the light's RGB color)
    """
    glBegin(GL_TRIANGLES)
    
    for i, face in enumerate(faces):
        normal = normals[i]
        
        # Initialize color contribution to (0, 0, 0)
        total_color = np.array([0.0, 0.0, 0.0])

        # Loop over each light and accumulate the color intensity
        for light in lights:
            # Calculate the intensity contribution of the light based on the normal and orientation
            intensity = (np.dot(normal, light.orientation)/2 + 0.5) * light.intensity
            intensity = max(intensity, 0.0)  # Ensure the intensity is non-negative

            # Calculate the color contribution by multiplying intensity with light color
            color_contribution = intensity * light.color

            # Accumulate the total color
            total_color += color_contribution

        # Clip the final color to be between 0 and 1 for each RGB channel
        total_color = np.clip(total_color, 0.0, 1.0)

        # Set the final color for the face
        glColor3fv(total_color)

        # Render the face (triangle) by connecting its vertices
        for vertex_idx in face:
            glVertex3fv(vertices[vertex_idx])

    glEnd()


def load_lights_from_hdr(hdr_file_path, max_number=32, intensity_factor=1.0, color_tuner=1.0):
    """
    Load light sources from an HDR file using importance sampling,
    with normalized and clamped RGB values and intensity.
    
    Args:
        hdr_file_path (str): Path to the HDR file.
        max_number (int): Maximum number of lights to sample. Default is 32.
    
    Returns:
        list: A list of light.LightSource instances.
    """
    # Load HDR image
    hdr_image = cv2.imread(hdr_file_path, cv2.IMREAD_UNCHANGED)
    if hdr_image is None:
        raise ValueError(f"Failed to load HDR file: {hdr_file_path}")

    # Calculate luminance using Rec. 709 coefficients
    luminance = np.dot(hdr_image[..., :3], [0.2126, 0.7152, 0.0722])  # Luminance
    max_luminance = np.max(luminance)
    if max_luminance == 0:
        raise ValueError("The HDR image contains no visible luminance.")

    # Normalize luminance to [0, 1]
    luminance /= max_luminance

    # Normalize HDR color values to [0, 1] based on the maximum RGB value
    max_color_value = np.max(hdr_image)
    normalized_hdr_image = hdr_image / max_color_value  # Normalize entire image to 0–1

    # Clamp normalized HDR image to [0, 1]
    normalized_hdr_image = np.clip(normalized_hdr_image, 0, 1)

    # Compute cumulative distribution function (CDF) for importance sampling
    height, width = luminance.shape
    cdf = np.cumsum(luminance.ravel())
    cdf /= cdf[-1]

    # Sample light sources
    sampled_lights = []
    for _ in range(max_number):
        # Sample a random value and find corresponding pixel
        sample = np.random.uniform(0, 1)
        pixel_index = np.searchsorted(cdf, sample)
        y, x = divmod(pixel_index, width)

        # Calculate light direction (assuming HDR is in latitude-longitude format)
        theta = np.pi * (y / height)  # Latitude
        phi = 2 * np.pi * (x / width)  # Longitude
        direction = np.array([
            np.sin(theta) * np.cos(phi),  # x
            np.sin(theta) * np.sin(phi),  # y
            np.cos(theta)                 # z
        ])

        # Get clamped, normalized color and normalized intensity
        color = normalized_hdr_image[y, x, :3]
        intensity = luminance[y, x]  # Already normalized to [0, 1]

        # Create light source instance
        sampled_lights.append(
            light.LightSource(orientation= direction, color=color * color_tuner, intensity=intensity * intensity_factor)
        )

    return sampled_lights



class Scene:
    def __init__(self):
        self.Camera = Camera
        self.Object3D = Object3D
        self.LightSource = LightSource

        self.cameras = {}
        self.objects = {}
        self.lights = {}

        self.pixel_density = 10
        self.wireframe_color = (0, 0, 0)
        self.face_color = (1, 1, 1)
        self.distribution_roughness = 0.5
        self.geometry_roughness = 0.5
        self.fresnel_value = 0.04
        self.render_resolution = (420, 420)

        # Define the allowed modes
        self.allowed_modes = {'wireframe', 'solidface', 'lambert', 'PBR_solidface', 'GGX_Distribution_solidface', 'GGX_Geometry_solidface', 'schlick_fresnel', 'PBR'}
        # Define the allowed modes for GPUs
        self.allowed_gpu_modes = {'wireframe', 'solid', 'lambert', 'pbr'}


        self.cameras['_globalCamera'] = Camera()

    # GPU FUNCTIONS! DON'T TOUCH! =============================================================================

    def load_lights_from_hdr(self, hdr_file_path, max_number=32, intensity_factor=1.0, color_tuner=1.0):
        """
        Load light sources from an HDR file using importance sampling,
        with normalized and clamped RGB values and intensity.
        
        Args:
            hdr_file_path (str): Path to the HDR file.
            max_number (int): Maximum number of lights to sample. Default is 32.
        
        Returns:
            list: A list of light.LightSource instances.
        """
        # Load HDR image
        hdr_image = cv2.imread(hdr_file_path, cv2.IMREAD_UNCHANGED)
        if hdr_image is None:
            raise ValueError(f"Failed to load HDR file: {hdr_file_path}")
    
        # Calculate luminance using Rec. 709 coefficients
        luminance = np.dot(hdr_image[..., :3], [0.2126, 0.7152, 0.0722])  # Luminance
        max_luminance = np.max(luminance)
        if max_luminance == 0:
            raise ValueError("The HDR image contains no visible luminance.")
    
        # Normalize luminance to [0, 1]
        luminance /= max_luminance
    
        # Normalize HDR color values to [0, 1] based on the maximum RGB value
        max_color_value = np.max(hdr_image)
        normalized_hdr_image = hdr_image / max_color_value  # Normalize entire image to 0–1
    
        # Clamp normalized HDR image to [0, 1]
        normalized_hdr_image = np.clip(normalized_hdr_image, 0, 1)
    
        # Compute cumulative distribution function (CDF) for importance sampling
        height, width = luminance.shape
        cdf = np.cumsum(luminance.ravel())
        cdf /= cdf[-1]
    
        # Sample light sources
        sampled_lights = []
        for _ in range(max_number):
            # Sample a random value and find corresponding pixel
            sample = np.random.uniform(0, 1)
            pixel_index = np.searchsorted(cdf, sample)
            y, x = divmod(pixel_index, width)
    
            # Calculate light direction (assuming HDR is in latitude-longitude format)
            theta = np.pi * (y / height)  # Latitude
            phi = 2 * np.pi * (x / width)  # Longitude
            direction = np.array([
                np.sin(theta) * np.cos(phi),  # x
                np.sin(theta) * np.sin(phi),  # y
                np.cos(theta)                 # z
            ])
    
            # Get clamped, normalized color and normalized intensity
            color = normalized_hdr_image[y, x, :3]
            intensity = luminance[y, x]  # Already normalized to [0, 1]
    
            # Create light source instance
            sampled_lights.append(
                light.LightSource(orientation= direction, color=color * color_tuner, intensity=intensity * intensity_factor)
            )
    
        return sampled_lights


    def get_light_data(self):
        MAX_LIGHTS = 32  # Maximum number of lights
    
        # Initialize arrays for light properties
        light_colors = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)  # Shape: (MAX_LIGHTS, 3)
        light_intensities = np.zeros(MAX_LIGHTS, dtype=np.float32)  # Shape: (MAX_LIGHTS,)
        light_orientations = np.zeros((MAX_LIGHTS, 3), dtype=np.float32)  # Shape: (MAX_LIGHTS, 3)
    
        # Get the number of active lights
        num_lights = min(len(self.lights), MAX_LIGHTS)
    
        # Fill the light properties arrays
        for i, light in enumerate(list(self.lights.values())[:num_lights]):
            light_colors[i] = light.color.flatten()  # Assuming color is a (1, 3) numpy array
            light_intensities[i] = light.intensity  # Intensity is a scalar
            light_orientations[i] = light.orientation.flatten()  # Assuming orientation is a (1, 3) numpy array

        return light_colors, light_intensities, light_orientations, num_lights

    def import_lights_to_scene(self, lights):
        """
        Import lights into a scene and register them with unique names.

        Args:
            lights (list): List of light.LightSource instances.
        """
        for i, light in enumerate(lights):
            light_name = f"light_{i + 1}"  # Create a unique name for each light
            self.register_object(light, light_name)  # Register the light in the scene

    # =========================================================================================================
    
    def _compute_view_vector(self, P):
        V = -P  # Camera is at (0, 0, 0), so view vector is -P
        return V / np.linalg.norm(V)

    def register_object(self, obj, name):
        if isinstance(obj, self.Camera):
            self.cameras[name] = obj
        elif isinstance(obj, self.Object3D):
            self.objects[name] = obj
        elif isinstance(obj, self.LightSource):
            self.lights[name] = obj
        else:
            raise Exception('`register_object(name)` requires `name` to be subclass of `Camera`, `LightSource` or `Object3D`. Provided: ', obj)

    def _switch_to_global_camera(self, active_name='_globalCamera'):

        theta_x, theta_y, theta_z = (self.cameras[active_name].orientation[0],
                                     self.cameras[active_name].orientation[1],
                                     self.cameras[active_name].orientation[2])

        for name, cam in self.cameras.items():
            self.cameras[name].orientation[0] -= theta_x
            self.cameras[name].orientation[1] -= theta_y
            self.cameras[name].orientation[2] -= theta_z

        for name, obj in self.objects.items():
            self.objects[name].rotate(-theta_x, 'x')
            self.objects[name].rotate(-theta_y, 'y')
            self.objects[name].rotate(-theta_z, 'z')

        a_x, a_y, a_z = (self.cameras[active_name].location[0],
                         self.cameras[active_name].location[1],
                         self.cameras[active_name].location[2])

        for name, cam in self.cameras.items():
            self.cameras[name].location[0] -= a_x
            self.cameras[name].location[1] -= a_y
            self.cameras[name].location[2] -= a_z

        for name, obj in self.objects.items():
            self.objects[name].centroid[0] -= a_x
            self.objects[name].centroid[1] -= a_y
            self.objects[name].centroid[2] -= a_z

    def _switch_from_global_camera(self, active_name):

        a_x, a_y, a_z = (self.cameras[active_name].location[0],
                         self.cameras[active_name].location[1],
                         self.cameras[active_name].location[2])

        for name, cam in self.cameras.items():
            self.cameras[name].location[0] -= a_x
            self.cameras[name].location[1] -= a_y
            self.cameras[name].location[2] -= a_z

        for name, obj in self.objects.items():
            self.objects[name].centroid[0] -= a_x
            self.objects[name].centroid[1] -= a_y
            self.objects[name].centroid[2] -= a_z

        theta_x, theta_y, theta_z = (self.cameras[active_name].orientation[0],
                                     self.cameras[active_name].orientation[1],
                                     self.cameras[active_name].orientation[2])

        for name, cam in self.cameras.items():
            self.cameras[name].orientation[0] -= theta_x
            self.cameras[name].orientation[1] -= theta_y
            self.cameras[name].orientation[2] -= theta_z

        for name, obj in self.objects.items():
            self.objects[name].rotate(-theta_x, 'x')
            self.objects[name].rotate(-theta_y, 'y')
            self.objects[name].rotate(-theta_z, 'z')


    def set_active_camera(self, active_name):
        """Sets the active camera by name."""
        if active_name in self.cameras:
            self.active_camera = self.cameras[active_name]
            self._switch_to_global_camera()
            self._switch_from_global_camera(active_name)
        else:
            raise Exception(f'Camera "{active_name}" is not registered in the scene.')

    def get_active_camera(self):
        """Returns the active camera."""
        return self.active_camera

    def perspective_projection(self, vertices, focal_length=1.0, d=1.0):
        # Separate the x, y, z coordinates
        x = vertices[:, :, 0]  # (faces, 4) -> x coordinates
        y = vertices[:, :, 1]  # (faces, 4) -> y coordinates
        z = vertices[:, :, 2]  # (faces, 4) -> z coordinates (depth)

        # Avoid division by zero by adding a small epsilon to z + d
        epsilon = 1e-6

        # Compute the perspective projection for x and y
        x_proj = (focal_length * x) / (z + d + epsilon)
        y_proj = (focal_length * y) / (z + d + epsilon)

        # Create the projected vertices array (faces, 4, 2) -> Only x' and y'
        projected_vertices = np.stack((x_proj, y_proj), axis=-1)

        return projected_vertices

    def _prepare_faces_for_rendering(self, objects):
        """
        Prepares and sorts faces of 3D objects by their Z-axis depth for rendering.

        Args:
            objects (list): A list of 3DObject instances, each with vertices, faces, tangents, and material.

        Returns:
            list: A list of dictionaries, each representing a face, sorted by Z-depth.
        """
        face_data = []

        for obj_index, obj in enumerate(objects):
            vertices = obj.get_vertices()
            faces = obj.get_faces()
            tangents = obj.get_tangents()  # Tangents for each face in the order of faces
            material = obj.material
        
            # Get vertices for each face
            face_vertices = vertices[faces]  # Shape: (num_faces, 3, 3)
        
            # Iterate through each face
            for face_num, (face_verts, tangent) in enumerate(zip(face_vertices, tangents)):
                # Calculate the average Z-depth of the face to use for sorting
                z_depth = np.mean(face_verts[:, 2])
            
                # Get the UV map and texture for the current face
                try:
                    uv_map = material.get_uv_map(face_num)
                except:
                    uv_map = None
            
                # Store all the necessary data for sorting and rendering
                face_data.append({
                    'z_depth': z_depth,       # Z-axis depth for sorting
                    'vertices': face_verts,    # 3D coordinates of the face's vertices
                    'uv_map': uv_map,          # UV mapping for this face
                    'material': material,        # Texture of this face
                    'tangent': tangent,        # Tangent of this face
                    'object_index': obj_index  # Index of the object this face belongs to
                })

        # Sort the faces by Z-depth in descending order (farthest to closest)
        sorted_faces = sorted(face_data, key=lambda x: x['z_depth'], reverse=True)
        return sorted_faces

    def render_gpu (self, mode = 'wireframe'):
        print('Warning: GPU Mode is highly experimental and might contain multiple bugs!')

        if mode not in self.allowed_gpu_modes:
            raise ValueError(f"Invalid render mode '{mode}'. Allowed modes are: {', '.join(self.allowed_gpu_modes)}")
            
        # Initialize PyGame and OpenGL
        pygame.init()
        display = (800, 800)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        # Set up OpenGL perspective
        #gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        #glTranslatef(0.0, 0.0, -10.0) # Not sure why it is here...
        glEnable(GL_DEPTH_TEST)

        vertices, faces = self.objects['defaultCube'].get_vertices(), self.objects['defaultCube'].get_faces()
        normals = self.objects['defaultCube'].get_tangents()
        uv = self.objects['defaultCube'].uv
        light_ = list(self.lights.values())
        light_colors, light_intensities, light_orientations, num_lights = self.get_light_data()

        if mode is 'pbr':
            #vertices, faces = self.objects['defaultCube'].get_vertices(), self.objects['defaultCube'].get_faces()
            #normals = self.objects['defaultCube'].get_tangents()
            #diffuse_tex = self.objects['defaultCube'].get_material().get_diffuse_texture()
            #uv = self.objects['defaultCube'].uv
            light_colors, light_intensities, light_orientations, num_lights = self.get_light_data()

            # Shader Code for First Triangle
            vertex_shader = """
            varying vec2 tex_coords;
            varying vec3 frag_normal;
            varying vec3 frag_position_world;
            
            void main() {
                // Pass texture coordinates to the fragment shader
                tex_coords = gl_MultiTexCoord0.xy;
            
                // Pass the normal (transformed to eye space) to the fragment shader
                frag_normal = normalize(gl_NormalMatrix * gl_Normal);
            
                // Compute vertex position using the legacy matrix
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            
                // Calculate the world-space position of the vertex
                frag_position_world = (gl_ModelViewMatrix * gl_Vertex).xyz;
            }
            
            
            """
            
            
            fragment_shader = """
            
            #define MAX_LIGHTS 32
            
            // Texture samplers
            uniform sampler2D texture_sampler;
            uniform sampler2D normal_map_sampler;      // Normal map
            uniform sampler2D occlusion_map_sampler;   // Occlusion map
            uniform sampler2D metallic_map_sampler;    // Metallic map
            uniform sampler2D roughness_map_sampler;   // Roughness map
            
            // Lighting uniforms
            uniform vec3 light_colors[MAX_LIGHTS];      // Colors for each light
            uniform float light_intensities[MAX_LIGHTS]; // Intensities for each light
            uniform vec3 light_directions[MAX_LIGHTS];  // Directions for each light
            uniform int num_lights;                     // The number of active lights
            
            // Flags for optional maps
            uniform bool use_normal_map;      // If true, normal map is applied
            uniform bool use_occlusion_map;   // If true, occlusion map is applied
            uniform bool use_metallic_map;    // If true, metallic map is applied
            uniform bool use_roughness_map;   // If true, roughness map is applied
            
            // Varying variables from the vertex shader
            varying vec2 tex_coords;
            varying vec3 frag_normal;         // Normal from the vertex shader
            varying vec3 frag_position_world; // World-space position of the fragment
            
            
            // GGX Normal Distribution Function (NDF)
            float ggx_distribution(vec3 normal, vec3 half_vector, float roughness) {
                float roughness_squared = roughness * roughness;
            
                // Dot product of the normal and the halfway vector
                float NdotH = max(dot(normal, half_vector), 0.0);
            
                // Denominator of GGX NDF formula
                float denom = (NdotH * NdotH) * (roughness_squared - 1.0) + 1.0;
                denom = denom * denom;
            
                // GGX NDF
                float D = roughness_squared / (3.14159265 * denom);
            
                return D;
            }
            
            
            // GGX Geometry Function (G)
            float ggx_geometry(vec3 normal, vec3 view_dir, vec3 light_dir, float roughness) {
                // Half-vector
                vec3 half_vector = normalize(view_dir + light_dir);
            
                // Dot products
                float NdotV = max(dot(normal, view_dir), 0.0);
                float NdotL = max(dot(normal, light_dir), 0.0);
                float NdotH = max(dot(normal, half_vector), 0.0);
                float VdotH = max(dot(view_dir, half_vector), 0.0);
                float LdotH = max(dot(light_dir, half_vector), 0.0);
            
                // GGX geometry (G) function
                float roughness_squared = roughness * roughness;
                float k = roughness_squared / 2.0;
            
                // Geometry term G1 for view direction and light direction
                float G1 = NdotV / (NdotV * (1.0 - k) + k);
                float G2 = NdotL / (NdotL * (1.0 - k) + k);
            
                return G1 * G2;
            }
            
            float fresnel_schlick(vec3 view_dir, vec3 light_dir, vec3 normal, float metallic) {
                float F0 = 0.04;  // Base reflectance for non-metals
                float cos_theta = max(dot(view_dir, normal), 0.0);
                return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0) * metallic;
            }
            
            float cook_torrance_pbr(float D, float G, float F, vec3 N, vec3 V, vec3 L) {
                float NdotV = max(dot(N, V), 0.0);  // Normal to View
                float NdotL = max(dot(N, L), 0.0);  // Normal to Light
            
                if (NdotV == 0.0 || NdotL == 0.0) {
                    return 0.0;  // Avoid division by zero or unphysical results
                }
            
                return (D * G * F) / (4.0 * NdotV * NdotL);
            }
            
            void main() {
                // Set default values for metallic and roughness if maps are not used
                float metallic = texture2D(metallic_map_sampler, tex_coords).r;
                float roughness = texture2D(roughness_map_sampler, tex_coords).r;
            
                // Set normal from normal map if available
                vec3 normal = frag_normal;
            
                // Initialize view direction (camera at origin)
                vec3 view_dir = normalize(-frag_position_world);
            
                // Accumulate PBR values from all lights
                float accumulated_pbr = 0.0;
            
                for (int i = 0; i < MAX_LIGHTS; ++i) {
                    if (i >= num_lights) {
                        break; // Stop if we exceed the number of active lights
                    }
            
                    // Normalize light direction for current light
                    vec3 light_dir = normalize(light_directions[i]);
            
                    // Compute the halfway vector
                    vec3 half_vector = normalize(light_dir + view_dir);
            
                    // Calculate the GGX distribution, geometry term, and Fresnel factor
                    float D = ggx_distribution(normal, half_vector, roughness);
                    float G = ggx_geometry(normal, view_dir, light_dir, roughness);
                    float F = fresnel_schlick(view_dir, light_dir, normal, metallic);
            
                    // Calculate PBR value using Cook-Torrance for this light
                    float pbr_value = cook_torrance_pbr(D, G, F, normal, view_dir, light_dir);
            
                    // Scale by light intensity and add to the accumulator
                    accumulated_pbr += pbr_value * light_intensities[i];
                }
            
            
                // Normalize the interpolated normal from the vertex shader
                vec3 norm_frag_normal = normalize(frag_normal);
            
                // Apply the normal map if enabled
                if (use_normal_map) {
                    vec3 normal_map = texture2D(normal_map_sampler, tex_coords).rgb;
                    normal_map = normalize(normal_map * 2.0 - 1.0); // Convert from [0, 1] to [-1, 1]
                    normal_map *= 10.0;
                    norm_frag_normal = normalize(norm_frag_normal + normal_map);
                }
            
                // Initialize the fragment color contribution from lighting
                vec3 light_contribution = vec3(0.0);
            
                // Iterate over all lights
                for (int i = 0; i < num_lights; i++) {
                    vec3 norm_light_dir = normalize(light_directions[i]);
                    
                    // Calculate the diffuse shading term
                    float diffuse_intensity = max(dot(norm_frag_normal, norm_light_dir), 0.0);
            
                    // Add this light's contribution
                    light_contribution += light_colors[i] * light_intensities[i] * diffuse_intensity;
                }
            
                // Sample the occlusion map if enabled
                float occlusion = use_occlusion_map 
                                  ? texture2D(occlusion_map_sampler, tex_coords).r 
                                  : 1.0;
            
                // Sample the base texture
                vec4 tex_color = texture2D(texture_sampler, tex_coords);
            
                // Final color calculation: texture color modulated by lighting and occlusion
                vec3 final_color_base = tex_color.rgb * light_contribution * occlusion;
            
                // Set the output color to the grayscale value (visualizing the accumulated PBR reflection term)
                gl_FragColor = vec4(vec3(accumulated_pbr) + final_color_base, 1.0);
            
            }
            
            void main_() {
                // Set default values for metallic and roughness if maps are not used
                float metallic = texture2D(metallic_map_sampler, tex_coords).r;
                float roughness = texture2D(roughness_map_sampler, tex_coords).r;
            
                // Set normal from normal map if available
                vec3 normal = frag_normal;
            
                // Apply normal map if enabled
                if (use_normal_map) {
                    vec3 normal_map = texture2D(normal_map_sampler, tex_coords).rgb;
                    normal = normalize(normal_map * 2.0 - 1.0); // Convert [0, 1] to [-1, 1]
                }
            
                // Initialize view direction and world-space normal
                vec3 view_dir = normalize(-frag_position_world);
            
                // Accumulate PBR and diffuse contributions
                vec3 accumulated_specular = vec3(0.0);
                vec3 accumulated_diffuse = vec3(0.0);
            
                for (int i = 0; i < MAX_LIGHTS; ++i) {
                    if (i >= num_lights) break;
            
                    // Normalize light direction for current light
                    vec3 light_dir = normalize(light_directions[i]);
            
                    // Compute the halfway vector
                    vec3 half_vector = normalize(light_dir + view_dir);
            
                    // Calculate PBR terms
                    float D = ggx_distribution(normal, half_vector, roughness);
                    float G = ggx_geometry(normal, view_dir, light_dir, roughness);
                    float F = fresnel_schlick(view_dir, light_dir, normal, metallic);
            
                    // Calculate specular reflection using Cook-Torrance model
                    float pbr_value = cook_torrance_pbr(D, G, F, normal, view_dir, light_dir);
                    accumulated_specular += pbr_value * light_colors[i] * light_intensities[i];
            
                    // Calculate diffuse contribution
                    float diffuse_intensity = max(dot(normal, light_dir), 0.0);
                    accumulated_diffuse += light_colors[i] * light_intensities[i] * diffuse_intensity;
                }
            
                // Sample occlusion map if enabled
                float occlusion = use_occlusion_map ? texture2D(occlusion_map_sampler, tex_coords).r : 1.0;
            
                // Sample albedo (diffuse color) from the base texture
                vec3 albedo = texture2D(texture_sampler, tex_coords).rgb;
            
                // Combine diffuse and specular contributions with occlusion
                vec3 final_color = occlusion * ((1.0 - metallic) * accumulated_diffuse * albedo + accumulated_specular);
            
                // Output the final fragment color
                gl_FragColor = vec4(final_color, 1.0);
            }
            
            
            """
            
            # Compile Shader Programs
            shader_program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )


            texture_ids_list = []
            for idx, obj in enumerate(self.objects.values()):
                # Load Texture Image
                texture_image = obj.get_material().get_diffuse_texture()
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_image.shape[1], texture_image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, texture_image)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)
                
                # Load the normal map
                normal_map = obj.get_material().get_normal_texture()
                normal_map_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, normal_map_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, normal_map.shape[1], normal_map.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, normal_map)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)
                
                # Load the occlusion map
                occlusion_map = obj.get_material().get_ao_texture()
                occlusion_map_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, occlusion_map_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, occlusion_map.shape[1], occlusion_map.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, occlusion_map)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)
                
                # Load the Metallic Map
                metallic_map = obj.get_material().get_metallic_texture()
                metallic_map_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, metallic_map_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, metallic_map.shape[1], metallic_map.shape[0], 0, GL_RED, GL_UNSIGNED_BYTE, metallic_map)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)
                
                # Load the Roughness Map
                roughness_map = obj.get_material().get_roughness_texture()
                roughness_map_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, roughness_map_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, roughness_map.shape[1], roughness_map.shape[0], 0, GL_RED, GL_UNSIGNED_BYTE, roughness_map)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)

                texture_ids_list.append((texture_id, normal_map_id, occlusion_map_id, metallic_map_id, roughness_map_id))
                
            # Initial camera position and rotation angles
            camera_pos = [0, 0, -5]  # Initial position of the camera (x, y, z)
            rotation_angles = [0, 0]  # [pitch, yaw] angles for rotation (initially set to 0)
            
            # Movement speed and rotation speed
            move_speed = 0.1
            rotate_speed = 2
            
            gluPerspective(45, 1, 0.1, 50.0)
            glTranslatef(0.0, 0.0, -5.0) # Not sure why it is here...
            
            glEnable(GL_DEPTH_TEST)
            
            
            
            running = True
            # Main game loop
            while running:
                # Event Handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
            
                # Continuously check key states
                keys = pygame.key.get_pressed()
            
                # Move the camera based on key states
                if keys[pygame.K_w]:  # Move forward
                    camera_pos[2] += move_speed
                if keys[pygame.K_s]:  # Move backward
                    camera_pos[2] -= move_speed
                if keys[pygame.K_a]:  # Move left
                    camera_pos[0] -= move_speed
                if keys[pygame.K_d]:  # Move right
                    camera_pos[0] += move_speed
            
                # Rotation based on key states
                if keys[pygame.K_UP]:  # Rotate up (pitch)
                    rotation_angles[0] += rotate_speed
                if keys[pygame.K_DOWN]:  # Rotate down (pitch)
                    rotation_angles[0] -= rotate_speed
                if keys[pygame.K_LEFT]:  # Rotate left (yaw)
                    rotation_angles[1] += rotate_speed
                if keys[pygame.K_RIGHT]:  # Rotate right (yaw)
                    rotation_angles[1] -= rotate_speed
            
              
                # Clear Screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
                for idx, obj in enumerate(self.objects.values()):
                    vertices = obj.get_vertices()
                    faces = obj.get_faces()
                    uv = obj.uv
                    normal = obj.get_tangents()
            
                    glPushMatrix()
                    glUseProgram(shader_program)
            
                    use_normal_map_location = glGetUniformLocation(shader_program, "use_normal_map")
                    use_occlusion_map_location = glGetUniformLocation(shader_program, "use_occlusion_map")
            
                    glUniform1i(use_normal_map_location, 1)  # True (or yes)
                    glUniform1i(use_occlusion_map_location, 1)  # False (or no)
            
            
                    t, n, o, m, r = texture_ids_list[idx]
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, t)
                    glUniform1i(glGetUniformLocation(shader_program, "texture_sampler"), 0)
            
            
                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, n)  # Normal map
                    glUniform1i(glGetUniformLocation(shader_program, "normal_map_sampler"), 1)
            
                    glActiveTexture(GL_TEXTURE2)
                    glBindTexture(GL_TEXTURE_2D, o)  # Occlusion map
                    glUniform1i(glGetUniformLocation(shader_program, "occlusion_map_sampler"), 2)
            
                    glActiveTexture(GL_TEXTURE3)
                    glBindTexture(GL_TEXTURE_2D, m)  # metallic map
                    glUniform1i(glGetUniformLocation(shader_program, "metallic_map_sampler"), 3)
            
                    glActiveTexture(GL_TEXTURE4)
                    glBindTexture(GL_TEXTURE_2D, r)  # roughness map
                    glUniform1i(glGetUniformLocation(shader_program, "roughness_map_sampler"), 4)
            
            
                    # Pass the num_lights uniform to the shader
                    glUseProgram(shader_program)
                    glUniform1i(glGetUniformLocation(shader_program, "num_lights"), num_lights)
            
                    # Pass the light properties to the shader
                    for i in range(num_lights):
                        glUniform3fv(glGetUniformLocation(shader_program, f"light_colors[{i}]"), 1, light_colors[i])
                        glUniform1f(glGetUniformLocation(shader_program, f"light_intensities[{i}]"), light_intensities[i])
                        glUniform3fv(glGetUniformLocation(shader_program, f"light_directions[{i}]"), 1, light_orientations[i])
            
                    # Apply camera movement and rotation
                    glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])  # Translate camera position
                    glRotatef(rotation_angles[0], 1, 0, 0)  # Rotate around the X-axis (pitch)
                    glRotatef(rotation_angles[1], 0, 1, 0)  # Rotate around the Y-axis (yaw)
            
                    for face_index, face in enumerate(faces):
                        glBegin(GL_TRIANGLES)
                        glNormal3fv(normals[face_index])
                        for vertex_index, uv_coord in zip(face, uv[face_index]):
                            glTexCoord2f(*uv_coord)  # Pass the UV coordinate (x, y) to the shader
                            glVertex3fv(vertices[vertex_index])  # Pass the vertex position to OpenGL
                        glEnd()
            
            
                    glUseProgram(0)
                    glPopMatrix()
            
                pygame.display.flip()
                pygame.time.wait(10)
            
            pygame.quit()



        if mode is 'wireframe':
            vertices, faces = self.objects['defaultCube'].get_vertices(), self.objects['defaultCube'].get_faces()
            normals = self.objects['defaultCube'].get_tangents()
            # Main rendering loop
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
        
                # Clear the screen and render the wireframe
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
                draw_wireframe(vertices, faces)
                for obj in self.objects.values():
                    vertices, faces = obj.get_vertices(), obj.get_faces()
                    draw_wireframe(vertices, faces)

                # Swap buffers to display the result
                pygame.display.flip()
                pygame.time.wait(10)

            pygame.quit()

        if mode is 'solid':
            light_ = list(self.lights.values())
            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            
            # Main rendering loop
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
        
                # Clear the screen and render the wireframe
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                for obj in self.objects.values():
                    vertices, faces = obj.get_vertices(), obj.get_faces()
                    normals = obj.get_tangents()
                    draw_solid_faces(vertices, faces, normals, light_)

                # Swap buffers to display the result
                pygame.display.flip()
                pygame.time.wait(10)

            pygame.quit()

        if mode is 'lambert':
            
            # Shader Code for First Triangle
            vertex_shader = """

            varying vec2 tex_coords;
            varying vec3 frag_normal;

            void main() {
                // Pass texture coordinates to the fragment shader
                tex_coords = gl_MultiTexCoord0.xy;

                // Pass the normal (transformed to eye space) to the fragment shader
                frag_normal = normalize(gl_NormalMatrix * gl_Normal);

                // Compute vertex position using the legacy matrix
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
            """

            
            fragment_shader = """

            #define MAX_LIGHTS 32

            uniform sampler2D texture_sampler;
            uniform sampler2D normal_map_sampler;   // Normal map
            uniform sampler2D occlusion_map_sampler; // Occlusion map

            uniform vec3 light_colors[MAX_LIGHTS];      // Colors for each light
            uniform float light_intensities[MAX_LIGHTS]; // Intensities for each light
            uniform vec3 light_directions[MAX_LIGHTS];  // Directions for each light
            uniform int num_lights;  // The number of active lights

            // Flags to check if normal map and occlusion map are provided
            uniform bool use_normal_map;   // If true, normal map is applied
            uniform bool use_occlusion_map; // If true, occlusion map is applied

            varying vec2 tex_coords;
            varying vec3 frag_normal;  // Normal from the vertex shader

            void main() {
                // Normalize the interpolated normal from the vertex shader
                vec3 norm_frag_normal = normalize(frag_normal);

                // Apply the normal map if enabled
                if (use_normal_map) {
                    vec3 normal_map = texture2D(normal_map_sampler, tex_coords).rgb;
                    normal_map = normalize(normal_map * 2.0 - 1.0); // Convert from [0, 1] to [-1, 1]
                    norm_frag_normal = normalize(norm_frag_normal + normal_map);
                }

                // Initialize the fragment color contribution from lighting
                vec3 light_contribution = vec3(0.0);

                // Iterate over all lights
                for (int i = 0; i < num_lights; i++) {
                    vec3 norm_light_dir = normalize(light_directions[i]);
        
                    // Calculate the diffuse shading term
                    float diffuse_intensity = max(dot(norm_frag_normal, norm_light_dir), 0.0);

                    // Add this light's contribution
                    light_contribution += light_colors[i] * light_intensities[i] * diffuse_intensity;
                }

                // Sample the occlusion map if enabled
                float occlusion = use_occlusion_map 
                                  ? texture2D(occlusion_map_sampler, tex_coords).r 
                                  : 1.0;

                // Sample the base texture
                vec4 tex_color = texture2D(texture_sampler, tex_coords);

                // Final color calculation: texture color modulated by lighting and occlusion
                vec3 final_color = tex_color.rgb * light_contribution * occlusion;

                // Output the final color with the texture alpha
                gl_FragColor = vec4(final_color, tex_color.a);
            }
            """

            # Compile Shader Programs
            shader_program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )

            texture_ids_list = []


            for idx, obj in enumerate(self.objects.values()):
                # Load Texture Image
                texture_image = imageio.imread("diffusion.png")
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_image.shape[1], texture_image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, texture_image)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)


                # Load the normal map
                normal_map = imageio.imread("normal.png")
                normal_map_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, normal_map_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, normal_map.shape[1], normal_map.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, normal_map)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)

                # Load the occlusion map
                occlusion_map = imageio.imread("ao.png")
                occlusion_map_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, occlusion_map_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, occlusion_map.shape[1], occlusion_map.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, occlusion_map)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)

                texture_ids_list.append((texture_id, normal_map_id, occlusion_map_id))

            
            running = True
            while running:
                # Event Handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                # Clear Screen
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                for idx, obj in enumerate(self.objects.values()):
                    vertices = obj.get_vertices()
                    faces = obj.get_faces()
                    uv = obj.uv
                    normal = obj.get_tangents()


                    glPushMatrix()
                    glUseProgram(shader_program)

                    # Assuming you have the shader program bound (e.g., glUseProgram(program))
                    use_normal_map_location = glGetUniformLocation(shader_program, "use_normal_map")
                    use_occlusion_map_location = glGetUniformLocation(shader_program, "use_occlusion_map")

                    # Set the flags to true (1 for true)
                    glUniform1i(use_normal_map_location, 1)
                    glUniform1i(use_occlusion_map_location, 1)

                    t, n, o = texture_ids_list[idx]

                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, t)
                    glUniform1i(glGetUniformLocation(shader_program, "texture_sampler"), 0)


                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, n)  # Normal map
                    glUniform1i(glGetUniformLocation(shader_program, "normal_map_sampler"), 1)

                    glActiveTexture(GL_TEXTURE2)
                    glBindTexture(GL_TEXTURE_2D, o)  # Occlusion map
                    glUniform1i(glGetUniformLocation(shader_program, "occlusion_map_sampler"), 2)


                    # Pass the num_lights uniform to the shader
                    glUseProgram(shader_program)
                    glUniform1i(glGetUniformLocation(shader_program, "num_lights"), num_lights)

                    # Pass the light properties to the shader
                    for i in range(num_lights):
                        glUniform3fv(glGetUniformLocation(shader_program, f"light_colors[{i}]"), 1, light_colors[i])
                        glUniform1f(glGetUniformLocation(shader_program, f"light_intensities[{i}]"), light_intensities[i])
                        glUniform3fv(glGetUniformLocation(shader_program, f"light_directions[{i}]"), 1, light_orientations[i])


                    for face_index, face in enumerate(faces):
                        glBegin(GL_TRIANGLES)
                        glNormal3fv(normals[face_index])
                        for vertex_index, uv_coord in zip(face, uv[face_index]):
                            glTexCoord2f(*uv_coord)  # Pass the UV coordinate (x, y) to the shader
                            glVertex3fv(vertices[vertex_index])  # Pass the vertex position to OpenGL
                        glEnd()


                    glUseProgram(0)
                    glPopMatrix()

                pygame.display.flip()
                pygame.time.wait(10)


            pygame.quit()


    def render(self, mode = 'wireframe'):
                
        # Check if the mode is valid
        if mode not in self.allowed_modes:
            raise ValueError(f"Invalid render mode '{mode}'. Allowed modes are: {', '.join(allowed_modes)}")

        focal_length, distance = self.active_camera.f, self.active_camera.d

        # Collect all objects and lights in the scene
        objects = list(self.objects.values())
        lights = list(self.lights.values())

        sorted_objects = self._prepare_faces_for_rendering(objects)

        # Extract sorted vertices and tangents
        sorted_vertices = np.array([face['vertices'] for face in sorted_objects])
        sorted_tangents = np.array([face['tangent'] for face in sorted_objects])
        sorted_face_positions = np.mean(sorted_vertices, axis=1, keepdims=True)

        if focal_length is not None:
            sorted_vertices = self.perspective_projection(sorted_vertices, focal_length, distance)[:, :, :2]
        else:
            sorted_vertices = sorted_vertices[:, :, :2]

        # Initialize light intensity accumulator
        sorted_light_intensity = np.zeros((sorted_tangents.shape[0], 3))

        # Calculate light intensity for each light source
        for light in lights:
            light_orientation = light.orientation.reshape(1, 3)  # Shape (1, 3)
            current_light_intensity = np.einsum('bi, ji -> bj', sorted_tangents, light_orientation)

            # Normalize intensity and accumulate light contributions
            current_light_intensity = current_light_intensity / 2 + 0.5  # Normalize to [0, 1]
            sorted_light_intensity += light.color * current_light_intensity

        # Clip the total light intensity to be in the range [0, 1]
        sorted_light_intensity = np.clip(sorted_light_intensity, 0, 1)

        # Find the min and max for x and y across all faces
        min_x = np.min(sorted_vertices[:, :, 0])
        max_x = np.max(sorted_vertices[:, :, 0])
        min_y = np.min(sorted_vertices[:, :, 1])
        max_y = np.max(sorted_vertices[:, :, 1])

        # Calculate x and y ranges
        (x_range, y_range) = (max_x - min_x, max_y - min_y)

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

        # Draw solid faces if requested
        if mode is 'solidface':
            for face, light_value in tqdm(zip(sorted_vertices, sorted_light_intensity)):
                polygon = patches.Polygon(face, closed=True, facecolor=light_value, alpha=1)
                ax.add_patch(polygon)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        # Draw wireframes if requested
        if mode is 'wireframe':
            for face, light_value in tqdm(zip(sorted_vertices, sorted_light_intensity)):
                polygon = patches.Polygon(face, closed=True, edgecolor=wireframe_color, facecolor=light_value, alpha=1)
                ax.add_patch(polygon)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        # Handle materials rendering if needed
        if mode is 'lambert':
            canvas = np.ones((self.render_resolution[0], self.render_resolution[1], 3))
            for face, obj, light_value in tqdm(zip(sorted_vertices, sorted_objects, sorted_light_intensity)):
                uv = obj['uv_map']
                texture = obj['material'].get_diffuse_texture()
                lambert_pipeline(canvas, face, uv, texture, light_value, ax, self.pixel_density)
            ax.imshow(canvas)

        if mode is 'PBR':
            canvas = np.ones((self.render_resolution[0], self.render_resolution[1], 3))
            for face, obj, face_position, normal in tqdm(zip(sorted_vertices, sorted_objects, sorted_face_positions, sorted_tangents)):                
                for light in lights:
                    spec_canvas = np.ones((self.render_resolution[0], self.render_resolution[1], 3))
                    L = light.orientation
                    V = self._compute_view_vector(face_position)
                    N = normal
                    H = L + V
                    H /= np.linalg.norm(H)
                    light_configs = (L, V, N, H)
                    PBR_material_pipeline(spec_canvas,
                                         face,
                                         obj,
                                         self.fresnel_value,
                                         light_configs,
                                         ax,
                                         pixel_density = self.pixel_density)
                    canvas += spec_canvas - 1
                    canvas = np.clip(canvas, 0, 1)
            canvas_diff = np.ones((self.render_resolution[0], self.render_resolution[1], 3))
            for face, obj, light_value in tqdm(zip(sorted_vertices, sorted_objects, sorted_light_intensity)):
                uv = obj['uv_map']
                texture = obj['material'].get_diffuse_texture()
                lambert_pipeline(canvas_diff, face, uv, texture, light_value, ax, self.pixel_density)
            final_canvas = np.clip(canvas + canvas_diff, 0, 1)
            ax.imshow(final_canvas)

        
        if mode is 'PBR_solidface':
            for face, face_tangents, face_position in tqdm(zip(sorted_vertices, sorted_tangents, sorted_face_positions)):
                face_color = 0
                for light in lights:
                    light_direction = light.orientation
                    view_direction = self._compute_view_vector(face_position)

                    H = light_direction + view_direction # (My view dir + light dir)/|My view dir + light dir = Half view
                    H /= np.linalg.norm(H)

                    face_color += cook_torrance_brdf(face_tangents, 
                                                     view_direction, 
                                                     light_direction, 
                                                     H, 
                                                     self.distribution_roughness, 
                                                     self.geometry_roughness, 
                                                     self.fresnel_value)

                # Now clip the values to [0, 1] and plot
                face_color = np.clip(face_color, 0, 1)
                polygon = patches.Polygon(face, closed=True, facecolor=(face_color, face_color, face_color), alpha=1)
                ax.add_patch(polygon)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        if mode is 'GGX_Distribution_solidface':
            for face, face_tangents, face_position in tqdm(zip(sorted_vertices, sorted_tangents, sorted_face_positions)):
                face_color = 0
                for light in lights:
                    light_direction = light.orientation
                    view_direction = self._compute_view_vector(face_position)

                    H = light_direction + view_direction # (My view dir + light dir)/|My view dir + light dir = Half view
                    H /= np.linalg.norm(H)

                    # Now light and my vectors gone, Normal view to be used here.
                    NdotH = np.dot(face_tangents.reshape(1, 3), H.reshape(1, 3).T) # Negate face_tangents if that doesn't work, transposing too to see if that works

                    face_color += ggx_distribution(NdotH, self.distribution_roughness)

                # Now clip the values to [0, 1] and plot
                face_color = np.clip(face_color, 0, 1)
                polygon = patches.Polygon(face, closed=True, facecolor=(face_color[0, 0], face_color[0, 0], face_color[0, 0]), alpha=1)
                ax.add_patch(polygon) 
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        if mode is 'GGX_Geometry_solidface':
            for face, face_tangents, face_position in tqdm(zip(sorted_vertices, sorted_tangents, sorted_face_positions)):
                face_color = 0
                for light in lights:
                    light_direction = light.orientation
                    view_direction = self._compute_view_vector(face_position)

                    face_color += ggx_geometry_full(face_tangents, view_direction, light_direction, self.geometry_roughness)

                # Now clip the values to [0, 1] and plot
                face_color = np.clip(face_color, 0, 1)

                polygon = patches.Polygon(face, closed=True, facecolor=(face_color, face_color, face_color), alpha=1)
                ax.add_patch(polygon)    
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        if mode is 'schlick_fresnel':
            for face, face_tangents, face_position in tqdm(zip(sorted_vertices, sorted_tangents, sorted_face_positions)):
                face_color = 0
                for light in lights:
                    light_direction = light.orientation
                    view_direction = self._compute_view_vector(face_position)

                    H = light_direction + view_direction # (My view dir + light dir)/|My view dir + light dir = Half view
                    H /= np.linalg.norm(H)
                    
                    face_color += fresnel_schlick(H, face_tangents, self.fresnel_value)

                # Now clip the values to [0, 1] and plot
                face_color = np.clip(face_color, 0, 1)
                polygon = patches.Polygon(face, closed=True, facecolor=(face_color, face_color, face_color), alpha=1)
                ax.add_patch(polygon)  
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Surface Plot from Vertices')

        # Finalize the plot
        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio square
        return fig
