import numpy as np

def ggx_geometry_full(N, V, L, alpha):
    """
    Computes the full GGX geometry function G(V, L, alpha) for both view and light directions.

    Parameters:
    - N (np.array): Normal vector of the surface (3D vector).
    - V (np.array): View vector (direction toward the viewer, 3D vector).
    - L (np.array): Light vector (direction toward the light source, 3D vector).
    - alpha (float): Roughness parameter (0 < alpha <= 1).

    Returns:
    - G (float): The full GGX geometry term for the given directions.
    """
    def G1(N, X, alpha):
        # Dot product of normal and direction vector
        N_dot_X = np.dot(N, X.T)
        
        # Ensure N_dot_X is non-negative
        N_dot_X = max(N_dot_X, 0.0)
        
        # Compute the single-direction GGX geometry term
        numerator = 2 * N_dot_X
        denominator = N_dot_X + np.sqrt(alpha**2 + (1 - alpha**2) * (N_dot_X ** 2))
        return numerator / denominator

    # Compute G1 for both view and light directions
    G_V = G1(N, V, alpha)
    G_L = G1(N, L, alpha)
    
    # The full GGX geometry term is the product of G1 for both directions
    G = G_V * G_L
    
    return G
