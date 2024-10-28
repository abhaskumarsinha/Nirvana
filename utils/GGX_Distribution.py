import numpy as np

def ggx_distribution(NdotH, alpha):
    """
    GGX Normal Distribution Function (NDF)
    
    Parameters:
    - NdotH: Cosine of the angle between normal (N) and half-vector (H)
    - alpha: Roughness parameter, where 0 is perfectly smooth and 1 is very rough
    
    Returns:
    - D: The GGX distribution value for the given angle and roughness
    """
    alpha2 = alpha * alpha
    NdotH2 = NdotH * NdotH
    
    # GGX Distribution formula
    denom = NdotH2 * (alpha2 - 1.0) + 1.0
    D = alpha2 / (np.pi * denom * denom)
    return D

