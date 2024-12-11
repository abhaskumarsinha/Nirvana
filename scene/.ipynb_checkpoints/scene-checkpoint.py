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

from Nirvana.camera.camera import *
from Nirvana.lights.light import *
from Nirvana.objects.base import *
from Nirvana.utils import *

# Move the function below to a different file
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
        self.allowed_gpu_modes = {'wireframe', 'solid', 'lambert'}


        self.cameras['_globalCamera'] = Camera()
    
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

        if mode not in self.allowed_gpu_modes:
            raise ValueError(f"Invalid render mode '{mode}'. Allowed modes are: {', '.join(self.allowed_gpu_modes)}")
            
        # Initialize PyGame and OpenGL
        pygame.init()
        display = (800, 800)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        # Set up OpenGL perspective
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -10.0) # Not sure why it is here...

        vertices, faces = self.objects['defaultCube'].get_vertices(), self.objects['defaultCube'].get_faces()
        normals = self.objects['defaultCube'].get_tangents()
        light_ = list(self.lights.values())


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