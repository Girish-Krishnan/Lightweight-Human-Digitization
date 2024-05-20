import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def load_and_capture_mesh(mesh_path, screenshot_path):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError("No mesh data loaded.")

    # Create a visualizer
    o3d.visualization.draw_geometries([mesh])

def display_image(image_path):
    # Load the image using PIL
    img = Image.open(image_path)

    # Display the image using matplotlib
    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.show()

# Example usage
mesh_path = 'mesh_5.ply'  # Path to your mesh file
screenshot_path = 'screenshot.png'  # Path where the screenshot will be saved

load_and_capture_mesh(mesh_path, screenshot_path)
