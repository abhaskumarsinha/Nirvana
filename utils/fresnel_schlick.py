import numpy as np

def fresnel_schlick(H, V, F0):
    """
    Computes the Fresnel reflectance using Schlick's approximation.

    Parameters:
    - H (np.array): Half-vector (3D vector).
    - V (np.array): View direction vector (3D vector).
    - F0 (float or np.array): Reflectance at normal incidence, depending on the material properties.

    Returns:
    - Fresnel reflectance (float or np.array): The Fresnel reflectance at the given angle.
    """
    # Compute the cosine of the angle between H and V
    cos_theta = np.dot(H, V.T)
    
    # Apply Schlick's approximation for the Fresnel term
    return F0 + (1 - F0) * (1 - cos_theta) ** 5
