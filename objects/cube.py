from Nirvana.objects.base import *
import numpy as np

class Cube(Object3D):
    def __init__(self, x=0, y=0, z=5, edge_length=1):
        # Save the position as the centroid of the cube
        self.edge_length = edge_length

        # Generate vertices at the origin
        vertices = self._generate_vertices()

        # Generate faces (this function would remain the same)
        faces = self._generate_faces()

        # Generate UV Map for the materials
        uv_map = self._generate_uv_map()

        # Initialize the Cube by passing vertices and faces to the base class
        super().__init__(vertices, faces, uv_map)

    def _generate_vertices(self):
        # Half the edge length for calculating the offsets from the center
        half_edge = self.edge_length / 2

        # Vertices generated around the origin (0, 0, 0)
        return np.array([
            [-half_edge, -half_edge, -half_edge],  # V0
            [-half_edge, -half_edge, half_edge],   # V1
            [-half_edge, half_edge, -half_edge],   # V2
            [-half_edge, half_edge, half_edge],    # V3
            [half_edge, -half_edge, -half_edge],   # V4
            [half_edge, -half_edge, half_edge],    # V5
            [half_edge, half_edge, -half_edge],    # V6
            [half_edge, half_edge, half_edge]      # V7
        ])

    def _generate_faces(self):
        # Define the 6 faces of the cube with correct CCW vertex ordering
        return [
            [0, 2, 3, 1],  # Left face  (CCW when viewed from outside)
            [4, 6, 7, 5],  # Right face (CCW when viewed from outside)
            [0, 4, 5, 1],  # Bottom face (CCW when viewed from outside)
            [2, 6, 7, 3],  # Top face (CCW when viewed from outside)
            [0, 4, 6, 2],  # Front face (CCW when viewed from outside)
            [1, 3, 7, 5],  # Back face (CCW when viewed from outside)
        ]

    def _generate_uv_map(self):
        return np.array([
            [[0, 1], [0, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [0, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [0, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [0, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [0, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 1]],
            [[0, 1], [0, 0], [1, 0]],
            [[1, 0], [0, 1], [1, 1]],
        ])
