import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

from Nirvana.camera.camera import *
from Nirvana.lights.light import *
from Nirvana.objects.base import *
from Nirvana.utils import *

class Scene:
    def __init__(self):
        self.Camera = Camera
        self.Object3D = Object3D
        self.LightSource = LightSource

        self.cameras = {}
        self.objects = {}
        self.lights = {}

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
                uv_map = material.get_uv_map(face_num)
            
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

    def render(self, show_wireframes=True,
               wireframe_color=(0, 0, 0),
               draw_solid_faces=True,
               render_materials=False):

        focal_length, distance = self.active_camera.f, self.active_camera.d

        # Collect all objects and lights in the scene
        objects = list(self.objects.values())
        lights = list(self.lights.values())

        # Get vertices, tangents and object indices for all objects
        all_vertices = np.concatenate([obj.get_vertices()[obj.get_faces()] for obj in objects], axis=0)
        all_tangents = np.concatenate([obj.get_tangents() for obj in objects], axis=0)

        sorted_objects = self._prepare_faces_for_rendering(objects)
        print(sorted_objects)


        # Calculate the average Z-coordinate for each face
        avg_z_coordinates = np.mean(all_vertices[:, :, 2], axis=1)

        # Apply perspective projection if focal length is provided
        if focal_length is not None:
            projected_coordinates = self.perspective_projection(all_vertices, focal_length, distance)
        else:
            projected_coordinates = all_vertices  # Use original vertices if no projection

        # Keep only the x and y coordinates
        vertex_2d = projected_coordinates[:, :, :2]

        # Sort indices based on the average Z-coordinate (from farthest to nearest)
        sorted_indices = np.argsort(avg_z_coordinates)[::-1]

        # Reorder vertices and tangents based on sorted indices
        # sorted_vertices = vertex_2d[sorted_indices]
        # sorted_tangents = all_tangents[sorted_indices]

        # Extract sorted vertices and tangents
        sorted_vertices = np.array([face['vertices'] for face in sorted_objects])
        sorted_tangents = np.array([face['tangent'] for face in sorted_objects])

        print('sorted vertices shape ', sorted_vertices.shape) # Shape (24, 3, 3)
        print('sorted tangents shape ', sorted_tangents.shape) # Shape (24, 3)

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

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

        # Draw solid faces if requested
        if draw_solid_faces:
            for face, light_value in zip(sorted_vertices, sorted_light_intensity):
                polygon = patches.Polygon(face, closed=True, facecolor=light_value, alpha=1)
                ax.add_patch(polygon)

        # Draw wireframes if requested
        if show_wireframes:
            for face, light_value in zip(sorted_vertices, sorted_light_intensity):
                polygon = patches.Polygon(face, closed=True, edgecolor=wireframe_color, facecolor=light_value, alpha=1)
                ax.add_patch(polygon)

        # Handle materials rendering if needed
        if render_materials:
            # Find a way to match a 3D Face to the exact 2D map indices of that object to which that 3D face belongs.
            raise NotImplementedError('Material rendering is a work in progress!')

        # Set plot limits and labels
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Surface Plot from Vertices')

        # Finalize the plot
        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio square
        return fig
