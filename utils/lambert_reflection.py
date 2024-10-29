import numpy as np
import matplotlib.pyplot as plt

from Nirvana.utils.barycentric_function import *

def lambert_pipeline(face, 
                     uv, 
                     texture, 
                     light_value, 
                     canvas, 
                     render_resolution, 
                     face_range):

    print('face values: ', face)
    print('face range: ', face_range)
    canvas[0, 0] = (0, 0, 0)
