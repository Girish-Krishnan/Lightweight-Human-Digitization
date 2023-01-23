import bpy
import os

# Set the directory where the .ply files are stored
directory = './reconstructed_video/'

# Set the output file path
output_file = os.path.abspath('./reconstructed_video/3d_video.blend')

# Get a list of all .ply files in the directory
ply_files = [f for f in os.listdir(directory) if f.endswith('.ply')]

# Define a key function to extract the frame number from the file name
def get_frame_number(file_name):
    # Split the file name by underscores
    parts = file_name.split('_')
    parts = parts[1].split('.')
    # Return the second part as an integer
    return int(parts[0])

# Sort the list of files using the key function
ply_files = sorted(ply_files, key=get_frame_number)

# Delete the default cube
bpy.ops.object.delete()

# Set the first frame of the animation
bpy.context.scene.frame_start = 1

# Set the frame rate of the animation
bpy.context.scene.render.fps = 60

# Set the number of frames for each point cloud
num_frames = 1

# Create an empty list to store the imported meshes
meshes = []

# Iterate through each .ply file
for i, ply_file in enumerate(ply_files):
    # Import the .ply file as a mesh
    bpy.ops.import_mesh.ply(filepath=os.path.join(directory, ply_file))
    
    # Get a reference to the imported mesh
    mesh = bpy.context.selected_objects[0]
    
    # Add the mesh to the list of meshes
    meshes.append(mesh)

# Set the current frame to the first frame
bpy.context.scene.frame_set(1)

# Set the visibility of the first mesh to be on
meshes[0].hide_viewport = False

# Set the keyframe for the visibility property
meshes[0].keyframe_insert(data_path='hide_viewport')

# Iterate through each frame and set the visibility of the correct mesh to be on
for i in range(1, len(meshes)):
    for j in range(num_frames * i + 1, num_frames * (i + 1) + 1):
        bpy.context.scene.frame_set(j)
        meshes[i-1].hide_viewport = True
        meshes[i-1].keyframe_insert(data_path='hide_viewport')
        meshes[i].hide_viewport = False
        meshes[i].keyframe_insert(data_path='hide_viewport')

# Set the end frame of the animation
bpy.context.scene.frame_end = num_frames * len(ply_files)

if os.path.exists(output_file):
    os.remove(output_file)

# Save the animation as a .blend file
bpy.ops.wm.save_as_mainfile(filepath=output_file)
