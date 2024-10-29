import numpy as np
import matplotlib.pyplot as plt

from Nirvana.utils.barycentric_function import *

def lambert_pipeline(face, 
                     uv, 
                     texture, 
                     light_value, 
                     canvas, 
                     self.render_resolution, 
                     face_range):


    canvas[5, 5] = (0, 0, 0)
