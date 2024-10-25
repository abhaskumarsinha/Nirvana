import numpy as np

class LightSource:
    def __init__(self, location=(0, 0, 0), orientation=(0, 0, 0), intensity=0.5, color=(1, 1, 1), epsilon=0.001):
        self.location = np.array(location)
        
        # Normalize color to the range [0, 1] if it's in [0, 255]
        self.color = np.array(color, dtype=float)
        if np.any(self.color > 1.0):  # Check if any value is greater than 1, assuming [0, 255] range
            self.color = self.color / 255.0  # Normalize to [0, 1]

        self.intensity = intensity
        
        # Calculate norm of the orientation vector
        norm = np.linalg.norm(orientation)
        self.orientation = np.array(orientation) / (norm + epsilon) * intensity
