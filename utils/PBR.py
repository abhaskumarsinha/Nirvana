import numpy as np

from Nirvana.utils.GGX_Distribution import *
from Nirvana.utils.GGX_Geometry import *
from Nirvana.utils.fresnel_schlick import *

from Nirvana.config.config import *


def cook_torrance_brdf(N, V, L, H, distribution_roughness, geometry_roughness, F0):
    """
    Calculate the Cook-Torrance BRDF.
    
    Parameters:
        N: The surface normal vector (3D vector).
        V: The direction vector to the viewer (3D vector).
        L: The direction vector of the incoming light (3D vector).
        H: The half angle between view and norml vector (3D vector).
        
        distribution_roughness: The roughness parameter for distribution (alpha).
        geometry_roughness: The roughness parameter for geometry (alpha).
        F0: The reflectance at normal incidence (typically, the material's reflectance).

    Returns:
        The computed BRDF value.
    """
    
    NdotH = np.dot(N, H.T)
    
    # Calculate D, G, F
    D_value = ggx_distribution(NdotH, distribution_roughness).item()  # Microfacet distribution function
    G_value = ggx_geometry_full(N, V, L, geometry_roughness)  # Geometric attenuation function
    F_value = fresnel_schlick(H, V, F0)  # Fresnel function

    # Calculate the BRDF
    denominator = 4 * np.dot(light_dir, normal.T) * np.dot(view_dir, normal.T)
    
    # Avoid division by zero
    if denominator == 0:
        return epsilon
    
    brdf_value = (D_value * G_value * F_value) / denominator
    
    return brdf_value.item()
