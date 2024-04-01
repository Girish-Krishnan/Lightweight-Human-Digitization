import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

mesh = o3d.io.read_triangle_mesh("3_rs-computed.ply")

def setup_scene_and_view():
    # Initialize the application and create a window
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window("Open3D", width=1024, height=768)

    # Create a SceneWidget to display the 3D scene
    widget = o3d.visualization.gui.SceneWidget()
    widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    # Define the material properties
    material = o3d.visualization.rendering.MaterialRecord()
    material.base_color = [0.7, 0.7, 0.7, 1.0]  # RGBA
    material.shader = "defaultLit"  # Use a lit shader suitable for lighting effects
    material.base_reflectance = 0.0  # Set a low reflectance to make the object appear matte
    material.base_roughness = 0.7  # Set a medium roughness to make the object appear matte
    material.base_metallic = 0.0  # Set a low metallic value to make the object appear matte
    print(dir(material))

    # Add the mesh with the defined material to the scene
    widget.scene.add_geometry("human_model", mesh, material)

    # Setup lighting - using Open3DScene's built-in methods
    # This example adds a directional light; you can adjust direction, color, and intensity as needed
    #widget.scene.scene.add_directional_light("main_light", [1, 1, 1], [1, 1, 1], 2.0)

    # Add a point light to the scene
    # widget.scene.scene.add_point_light([0, 0, 0], [1, 1, 1], 1.0, 0.0, True)

    print(dir(widget.scene.scene))

    # Add the widget to the window and set up the layout
    window.add_child(widget)

    # For visualization, allow the user to move mouse to rotate, zoom, and pan the view
    # The widget's camera is accessible and can be manipulated directly

    # Run the application
    app.run()

setup_scene_and_view()
#if not mesh.is_triangle_mesh():
#    mesh = mesh.triangulate()
is_watertight = mesh.is_watertight()

# Print the watertight status
print(f"Mesh is {'watertight' if is_watertight else 'not watertight'}.")

# Export to STL in binary format
# Compute the normals of the mesh and orient to consistent tangent plane
mesh.compute_vertex_normals()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
o3d.io.write_triangle_mesh("output_file_path.stl", mesh, write_ascii=False)

print("Mesh exported to STL format.")