import numpy as np

class LightSource:
    def __init__(self, location=(0, 0, 0), orientation=(0, 0, 0), intensity=0.5, color=(1, 1, 1), epsilon=0.001):
        self.location = np.array(location)
        self.color = np.array(color)
        self.intensity = intensity
        self.color = np.array(color)

        # Calculate norm of the orientation vector
        norm = np.linalg.norm(orientation)

        self.orientation = np.array(orientation)/(norm + epsilon) * intensity
