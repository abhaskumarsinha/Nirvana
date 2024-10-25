import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio


class Scene:
    def __init__(self):
        self.cameras = {}
        self.objects = {}
        self.lights = {}

        self.cameras['_globalCamera'] = Camera()
    
    def _compute_view_vector(self, P):
        V = -P  # Camera is at (0, 0, 0), so view vector is -P
        return V / np.linalg.norm(V)

    def register_object(self, obj, name):
        if isinstance(obj, Camera):
            self.cameras[name] = obj
        elif isinstance(obj, Object3D):
            self.objects[name] = obj
        elif isinstance(obj, LightSource):
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


    def render(self, show_wireframes = True,
               wireframe_color = (0,0,0),
               draw_solid_face = True,
               render_materials = False):
        # TEMP!! THIS WOULD GET REMOVED IN NEAR FUTURE!!
        # texture = imageio.imread('texture.jpg') / 255.0
        # View direction (camera looking along Z-axis)
        V = np.array([0.0, 0.0, 1.0])

        focal_length, d = self.active_camera.f, self.active_camera.d

        objects = [obj for name, obj in scene.objects.items()]
        lights = [obj for name, obj in scene.lights.items()]

        coordinates = objects[0].get_vertices()[objects[0].get_faces()]
        tangents = objects[0].get_tangents()
        for i in range(1, len(objects)):
            coordinates = np.concatenate((coordinates, objects[i].get_vertices()[objects[i].get_faces()]), axis=0)
            tangents = np.concatenate((tangents, objects[i].get_tangents()), axis=0)

        # Step 1: Calculate the average Z-coordinate for each face
        z_avg = np.mean(coordinates[:, :, 2], axis=1)  # Shape: (num_faces,)

        # Apply prespective projection if applicable.
        if focal_length is not None:
            coordinates_post_projection = self.perspective_projection(coordinates, focal_length, d)

        # Drop the last axis (z-coordinates) to get shape (number of faces, 4, 2)
        vertices =  coordinates_post_projection[:, :, :2]  # This keeps only the x and y coordinates

        # Step 2: Sort indices based on the average Z-coordinate (from farthest to nearest)
        sorted_indices = np.argsort(z_avg)[::-1]

        # Step 3: Reorder the vertices and tangents based on the sorted indices
        sorted_vertices = vertices[sorted_indices]  # Sorted based on Z-buffering
        sorted_tangents = tangents[sorted_indices]
        sorted_face_pos = np.mean(coordinates[sorted_indices], axis=1)

        # Step 4: Recalculate light intensity with sorted tangents
        # Initialize sorted_light_intensity as zero for accumulation
        sorted_light_intensity = np.zeros((sorted_tangents.shape[0], 3))  # Shape: (num_faces, 1)

        for light in lights:
            # Calculate the light intensity for the current light
            light_orientation = light.orientation.reshape(1, 3)  # Shape (1, 3)
            current_light_intensity = np.einsum('bi, ji -> bj', sorted_tangents, light_orientation)

            # Normalize intensity and add to the total intensity
            current_light_intensity = current_light_intensity / 2 + 0.5  # Normalize to [0, 1]
            sorted_light_intensity += light.color * current_light_intensity  # Sum intensities from all lights

        # Clip the total light intensity to be in the range [0, 1]
        sorted_light_intensity = np.clip(sorted_light_intensity, 0, 1)

        # Step 1: Extract orientations
        orientations = np.array([light.orientation.reshape(1, 3) for light in lights])

        # Step 2: Compute the average orientation
        average_orientation = np.mean(orientations, axis=-1)


        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

        if draw_solid_face:
            # Loop through each face and plot it
            for face, light_value, tangents, face_pos in zip(sorted_vertices, sorted_light_intensity, sorted_tangents, sorted_face_pos):
                # Calculate face lighting values
                metallic = 0.3
                roughness = 0.9
                facecolor = cook_torrance_brdf(tangents,
                                               _compute_view_vector(face_pos),
                                               lights[0].orientation,
                                               np.array([0.8, 0.8, 0.8]), # Base color as white?
                                               metallic,
                                               roughness,
                                               lights[0].color)
                # Create a polygon patch for the current face
                polygon = patches.Polygon(face, closed=True, facecolor=(facecolor[0], facecolor[1], facecolor[2]), alpha=1)
                ax.add_patch(polygon)
        if show_wireframes:
            for face, light_value in zip(sorted_vertices, sorted_light_intensity):
                # Create a polygon patch for the current face
                polygon = patches.Polygon(face, closed=True, edgecolor=wireframe_color, facecolor=(light_value[0], light_value[1], light_value[2]), alpha=1)
                ax.add_patch(polygon)
        if render_materials:
            for face, light_value, tangent in zip(sorted_vertices, sorted_light_intensity, sorted_tangents):
                # UV value and Texture WILL GET REMOVED AFTER IMPLEMENTING MATERIALS!
                #render_textured_face(face, np.array([[0, 0], [1, 0], [0, 1]]), texture, light_value, ax, pixel_density = 15)
                metallic = 0.3
                roughness = 0.6

                render_textured_face(face,
                                     np.array([[0, 0], [1, 0], [0, 1]]),
                                     texture,
                                     light_value,
                                     tangent.reshape(1, 3),
                                     np.array([0, 0, -1]),
                                     light_direction = average_orientation,
                                     metallic = metallic,
                                     roughness = roughness,
                                     ax = ax,
                                     pixel_density = 10)

        # Set limits and labels
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Surface Plot from Vertices')

        # Show the plot
        plt.grid()
        plt.gca().set_aspect('equal', adjustable='box')  # Keep aspect ratio square
        return fig
