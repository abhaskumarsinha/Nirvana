class Camera:
    def __init__(self, location=(0, 0, 0), orientation=(0, 0, 0), f = None, d = None):

        self.default_location = list(location)
        self.default_orientation = list(orientation)
        self.location = list(location)
        self.orientation = list(orientation)

        self.f, self.d = f, d

    def reset(self):
        self.location = self.default_location
        self.orientation = self.default_orientation
        print("Camera reset to default values.")

    def __repr__(self):
        return f"Camera(location={self.location}, orientation={self.orientation})"
  
