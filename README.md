# Nirvana-3D
![nirvana-3d-high-resolution-logo-transparent](https://github.com/user-attachments/assets/ed1438f3-f74d-424c-b770-0ef1b60a0346)
##### [[`LOGO CREDITS`]](https://logo.com) [[`LICENSE`]](https://github.com/abhaskumarsinha/Nirvana-3D/blob/main/LICENSE) [[`Container Model Artist: jrgubric`]](https://free3d.com/user/jrgubric)
*Dispel the ignorance of the illusionary world; transcend the game engine!*: presenting a Game Engine written on Python, extending support across multiple GPU devices with 3D rendering, PBR Renderer and much more!
# The Engine is Currently WIP!! ⚠️

## Abstract
**Nirvana 3D** is an experimental 3D graphics and game engine written in Python, built to explore the fundamentals of 3D rendering and engine design. Designed initially for offline rendering, Nirvana 3D can import and render 3D OBJ models with basic materials and lighting. The engine currently focuses on educational purposes, providing a platform for understanding and developing essential 3D engine components before transitioning to real-time rendering.
### Features
- **3D Model Import**: Supports importing 3D OBJ models with UV mapping for texture and material rendering.
- **Rendering Techniques**: Renders scenes using solid face rendering and Lambert shading, simulating diffuse lighting.
- **Camera Systems**: Supports both orthographic and perspective cameras, offering flexibility for various visual perspectives.
- **Basic Lighting**: Implements sunlight-based directional lighting to simulate environmental lighting.
- **Transformations**: Supports transformations like translation, rotation, and scaling for models and objects within the scene.
- **Visualization**: Renders directly to Python’s `matplotlib`, allowing easy visualization of 3D scenes even in environments without a dedicated graphics API.
### Future Development
Nirvana 3D is in its early stages, with plans to enhance performance and extend its capabilities, including:

- **GPU Acceleration**: Shift from CPU-bound rendering to GPU-based processing for real-time rendering performance.
- **Physically Based Rendering (PBR)**: Introduce advanced shading models and PBR for realistic lighting and material rendering.
- **Additional Lighting Types**: Expand beyond directional lighting to support point, spot, and area lights.
- **User Controls**: Add advanced interactivity with camera and object manipulation controls.
- **Collision Detection**: Incorporate collision detection for basic physics interactions within the engine.
Nirvana 3D aims to eventually evolve into a versatile, Python-powered 3D engine for educational and experimental purposes. Contributions and suggestions are welcome to help shape its future!

## Screenshots
https://github.com/user-attachments/assets/a5e81ef1-5c15-4556-bc50-1b70789a2dce

*3D Models interacting with lights in Lambertian mode.*
![nirvana_renderer](https://github.com/user-attachments/assets/3cf0f478-010d-47b7-9864-de012877af18)
![nirvana_renderer (1)](https://github.com/user-attachments/assets/dec865a0-f39e-484b-945b-ffa407046d82)
*Rendering a 3D Cube with Nirvana-3D logo on it.*
![container_rendering](https://github.com/user-attachments/assets/f2173d4f-0cd1-47a6-9b93-fe6dfb07f883)
*A 3D rendering container Model imported in Nirvana and being rendered with solid face.*

https://github.com/user-attachments/assets/701018ba-7e21-4ce6-bd9a-06c432ac8cb2

*A 3D rendering video of a Container Model being rendered frame wise in Nirvana 3D.*
Model source: https://free3d.com/3d-model/container-169022.html
Artist Credits: [[`jrgubric`]](https://free3d.com/user/jrgubric)

## Getting Started

#### 1. Render a cube with a diffuse texture and lambert shading
```python
# Importing core classes for objects, lighting, camera, scene, and materials
from Nirvana.objects import *
from Nirvana.lights import *
from Nirvana.camera import *
from Nirvana.scene import *
from Nirvana.material import *

# Initialize a base material with a texture
nirvana_texture = BaseMaterial('./nirvana.png')

# Create a cube object and apply transformations
cube1 = cube.Cube()
cube1.rotate(10, 'x')               # Rotate cube by 10 degrees along the x-axis
cube1.rotate(10, 'y')               # Rotate cube by 10 degrees along the y-axis
cube1.triangulate()                 # Convert cube faces into triangles for rendering
cube1.calculate_tangents()          # Calculate tangents for proper texture mapping
cube1.set_material(nirvana_texture) # Apply the texture material to the cube

# Create light sources to illuminate the scene
sunlight = light.LightSource(orientation=(-0.5, -1, 0), color=(135, 206, 235))  # Primary sunlight with color
sunlight2 = light.LightSource(orientation=(0.5, -1, 0.5), color=(1, 1, 1), intensity=0.2)  # Secondary light with low intensity

# Set up the camera for the scene
camera_ = camera.Camera(d=10, f=30)  # Position the camera with specified distance and focal length

# Create and configure the scene
scene = scene.Scene()
scene.register_object(cube1, 'defaultCube')  # Add cube to the scene with an identifier
scene.register_object(sunlight, 'light')     # Register the main light source
scene.register_object(sunlight2, 'light2')   # Register secondary light source
scene.register_object(camera_, 'cam')        # Register camera in the scene
scene.set_active_camera('cam')               # Set the active camera for rendering

# Configure rendering settings and render the scene
scene.pixel_density = 50                      # Set pixel density for render resolution
scene.render('lambert').savefig('nirvana_renderer.jpg', dpi=500)  # Render in lambert shading mode and save as image
```

#### 2. Rendering an *.obj model file in solidface mode
```python
# Import core modules for camera, lights, materials, 3D objects, and scene setup
from Nirvana.camera import *
from Nirvana.lights import *
from Nirvana.material import *
from Nirvana.objects import *
from Nirvana.scene import *

import numpy as np  # Import NumPy for possible transformations or calculations

# Initialize 3D objects to represent containers in the scene
container = Object3D()          # First container object
container_2 = Object3D()        # Second container object

# Load 3D model and apply transformations to the first container
container.load_obj('Container.obj')    # Load OBJ file for the container model
container.translate(5, -50, 500)       # Position the container in 3D space
container.rotate(10, 'y')              # Rotate the container around the y-axis
container.calculate_tangents()         # Calculate tangents for texture mapping

# Load and transform a second instance of the container model
container_2.load_obj('Container.obj')  # Load the same OBJ file for second container
container_2.translate(150, -50, 300)   # Position second container differently
container_2.rotate(-10, 'y')           # Rotate the second container around the y-axis
container_2.calculate_tangents()       # Calculate tangents for consistent texture mapping

# Define lighting sources to illuminate the scene
sunlight = LightSource(orientation=(-0.5, -1, 0), color=(135, 206, 235), intensity=0.3)  # Main directional light
backlight = LightSource(orientation=(0.5, -1, 0), intensity=0.7)                        # Secondary light for depth

# Set up the camera for scene framing and perspective
camera = Camera(f=20, d=30)            # Define camera with focal length and distance

# Create the scene and register objects, lights, and camera
scene = Scene()
scene.register_object(container, 'container')       # Register first container with identifier
scene.register_object(container_2, 'sec_container') # Register second container
scene.register_object(sunlight, 'sunlight')         # Register main light source
scene.register_object(backlight, 'backlight')       # Register secondary light source
scene.register_object(camera, 'cam')                # Register camera in the scene

# Set the active camera for rendering
scene.set_active_camera('cam')

# Render the scene using solid face rendering and save the output image
scene.render('solidface').savefig('./container_rendering.jpg', dpi=500)  # Save high-resolution render
```

# References
- Akenine-Moller, Tomas, Eric Haines, and Naty Hoffman. Real-time rendering. AK Peters/crc Press, 2019.





