import numpy as np

from Nirvana.material.base import *

class Object3D:
    def __init__(self, vertices=None, faces=None, tangents=None, uv_map=None, material=None):
        """
        Initialize Object3D with vertices, faces, and optional material.

        :param vertices: List or array of 3D vertices of the object.
        :param faces: List or array of faces (each face is a list of vertex indices).
        :param tangents: List or array of tangents.
        :param uv_map: UV mapping data for the object.
        :param material: (Optional) A material object (instance of BaseMaterial or derived class).
        """
        self.MaterialClass = BaseMaterial
        self.centroid = np.mean(vertices, axis=0) if vertices is not None else np.zeros(3)
        self.vertices = np.array(vertices, dtype=np.float64) - self.centroid if vertices is not None else None
        self.faces = np.array(faces) if faces is not None else None
        self.tangents = tangents
        self.uv = uv_map
        self.material = material  # Optional material object

    def load_obj(self, file_path):
        vertices = []
        faces = []
        normals = []
        uv_coords = []

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                prefix, data = parts[0], parts[1:]

                if prefix == 'v':  # Vertex data
                    vertices.append([float(coord) for coord in data])
                elif prefix == 'vt':  # Texture coordinate data
                    uv_coords.append([float(coord) for coord in data])
                elif prefix == 'vn':  # Normal vector data
                    normals.append([float(coord) for coord in data])
                elif prefix == 'f':  # Face data
                    face = []
                    for vertex in data:
                        # Extract indices, or set to None if not specified
                        v_idx, vt_idx, vn_idx = (int(i) - 1 if i else None for i in vertex.split('/'))
                        face.append((v_idx, vt_idx, vn_idx))

                    # Filter out faces with any None values
                    if all(None not in vertex_tuple for vertex_tuple in face):
                        # Triangulate the face if it has more than 3 vertices
                        if len(face) > 3:
                            for i in range(1, len(face) - 1):
                                faces.append([face[0], face[i], face[i + 1]])
                        else:
                            faces.append(face)  # Keep as-is if it's already a triangle

        # Convert lists to numpy arrays for easy manipulation in the class
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces)  # This will now contain only triangles without None values
        normals = np.array(normals, dtype=np.float64)
        uv_coords = np.array(uv_coords, dtype=np.float64)

        # Pass the cleaned and triangulated data to the constructor
        self.__init__(vertices=vertices, faces=faces, tangents=normals, uv_map=uv_coords)




    def set_material(self, material):
        """
        Set the material for the 3D object.

        :param material: A material object (instance of BaseMaterial or derived class).
        """
        if not isinstance(material, self.MaterialClass):
            raise TypeError("Material must be an instance of BaseMaterial or its subclasses.")
        self.material = material
        if self.uv is not None:
            self.material.uv_map = self.uv

    def get_material(self):
        """
        Get the material associated with the 3D object.

        :return: The material object, or None if no material is assigned.
        """
        return self.material

    def get_vertices(self):
        return self.vertices + self.centroid

    def get_faces(self):
        return self.faces

    def calculate_tangents(self):
        """
        Calculate the tangents for triangular or quadrilateral faces.
        Returns tangents (ndarray) for each face.
        """
        faces = self.get_vertices()[self.get_faces()]

        # Check if faces have 3 or 4 vertices
        if faces.shape[1] == 3:  # Triangular faces
            edge1 = faces[:, 1] - faces[:, 0]
            edge2 = faces[:, 2] - faces[:, 0]
        elif faces.shape[1] == 4:  # Quadrilateral faces
            edge1 = faces[:, 1] - faces[:, 0]
            edge2 = faces[:, 2] - faces[:, 0]
        else:
            raise ValueError("Each face must have 3 or 4 vertices.")

        # Cross product of two edges for each face gives the normal
        tangents = np.cross(edge1, edge2)

        # Normalize the tangents to get unit vectors
        norm = np.linalg.norm(tangents, axis=1, keepdims=True)
        self.tangents = tangents / norm

    def get_tangents(self):
        return self.tangents

    def translate(self, dx, dy, dz):
        """
        Translate the object by (dx, dy, dz).
        """
        translation_matrix = np.array([dx, dy, dz])
        self.centroid += translation_matrix
        self.vertices += translation_matrix

    def triangulate(self):
        """
        Convert quadrilateral faces into triangular faces.
        """
        triangles = []
        for face in self.get_faces():
            # Split quadrilateral face into two triangles
            triangles.append([face[0], face[1], face[2]])  # Triangle 1
            triangles.append([face[0], face[2], face[3]])  # Triangle 2
        self.faces = np.array(triangles)

    def scale(self, scale_factor):
        """
        Scale the object relative to its centroid.
        """
        translated_vertices = self.vertices - self.centroid
        translated_vertices *= scale_factor
        self.vertices = translated_vertices + self.centroid

    def rotate(self, angle, axis, centroid=(0, 0, 0)):
        """
        Rotate the object around a given axis by the specified angle (in radians).
        """
        angle_rad = angle
        translated_vertices = self.vertices - self.centroid

        # Create rotation matrix for the given axis
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        # Rotate vertices and translate back to the original position
        rotated_vertices = np.dot(translated_vertices + centroid, rotation_matrix)
        self.vertices = rotated_vertices + self.centroid
