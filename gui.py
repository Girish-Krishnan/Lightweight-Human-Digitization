import tkinter as tk
from tkinter import filedialog, messagebox, IntVar, StringVar
import subprocess

def hardware_reset():
    try:
        subprocess.run(["python", "capture.py", "--hardware_reset"], check=True)
        messagebox.showinfo("Success", "Hardware reset successful.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Hardware reset failed.")

def select_output_dir():
    dir_name = filedialog.askdirectory()
    output_dir.set(dir_name)

def select_config_file():
    file_name = filedialog.askopenfilename(filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
    config_file.set(file_name)

def select_output_file():
    file_name = filedialog.askopenfilename(filetypes=(("PLY files", "*.ply"), ("All files", "*.*")))
    output_file.set(file_name)

def select_odom_file():
    file_name = filedialog.askopenfilename(filetypes=(("Log files", "*.log"), ("All files", "*.*")))
    odom_file.set(file_name)

def select_mesh_file():
    file_name = filedialog.askopenfilename(filetypes=(("PLY files", "*.ply"), ("All files", "*.*")))
    mesh_file.set(file_name)

def select_directory_for_output_file():
    dir_name = filedialog.askdirectory()
    if dir_name:  # Check if a directory was selected
        output_file.set(dir_name + "/")

def select_directory_for_odom_file():
    dir_name = filedialog.askdirectory()
    if dir_name:
        odom_file.set(dir_name + "/")

def select_directory_for_mesh_file():
    dir_name = filedialog.askdirectory()
    if dir_name:
        mesh_file.set(dir_name + "/")

def update_resolution(*args):
    res = resolution.get()
    w, h = res.split('x')
    width.set(w)
    height.set(h)

def validate_integer_input(input_str, min_val, max_val):
    try:
        # Convert the input to an integer
        val = int(input_str)
        # Check if the value is within the specified range
        if min_val <= val <= max_val:
            return True
        else:
            messagebox.showerror("Invalid Input", f"Value must be between {min_val} and {max_val}.")
            return False
    except ValueError:
        # The input is not a valid integer
        messagebox.showerror("Invalid Input", "Please enter a valid integer.")
        return False

def validate_file_extensions():
    if not config_file.get().endswith('.json'):
        messagebox.showerror("Invalid File", "Config file must be a .json file.")
        return False
    if not output_file.get().endswith('.ply'):
        messagebox.showerror("Invalid File", "Output file must be a .ply file.")
        return False
    if not odom_file.get().endswith('.log'):
        messagebox.showerror("Invalid File", "Odometry log file must be a .log file.")
        return False
    if not mesh_file.get().endswith('.ply'):
        messagebox.showerror("Invalid File", "Mesh file must be a .ply file.")
        return False
    return True  # All validations passed

def validate_warmup_frames():
    try:
        warmup_frames_value = int(warmup_frames.get())
        if 0 <= warmup_frames_value <= 10000:
            return True
        else:
            messagebox.showerror("Invalid Input", "Warmup Frames must be between 0 and 10000.")
            return False
    except ValueError:
        messagebox.showerror("Invalid Input", "Warmup Frames must be a valid integer.")
        return False

def validate_num_captures():
    try:
        num_captures_value = int(num_captures.get())
        if 1 <= num_captures_value <= 1000:
            return True
        else:
            messagebox.showerror("Invalid Input", "Number of Captures must be between 1 and 1000.")
            return False
    except ValueError:
        messagebox.showerror("Invalid Input", "Number of Captures must be a valid integer.")
        return False

def submit():
    if not validate_warmup_frames():
        return
    
    if not validate_num_captures():
        return

    if not validate_file_extensions():
        return
    
    cmd = [
        "python", "capture.py",
        "--output_dir", output_dir.get(),
        "-w", str(width.get()),
        "-ht", str(height.get()),
        "-f", str(fps.get()),
        "--warmup-frames", warmup_frames.get(),
        "-n", num_captures.get()
    ]
    if data_reset.get() == 1:
        cmd.append("--data_reset")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Capture failed.")
        return

    cmd = [
        "python", "combine_pcd.py",
        "--config_file", config_file.get(),
        "--output_file", output_file.get(),
        "--data_dir", output_dir.get(),
        "--odom_file", odom_file.get(),
        "--mesh_file", mesh_file.get()
    ]
    if save_individual.get() == 1:
        cmd.append("--save_individual")

    try:
        subprocess.run(cmd, check=True)
        messagebox.showinfo("Success", "Processing completed successfully.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Processing failed.")


# Tkinter setup with labels
root = tk.Tk()
root.title("Capture GUI")

root.tk_setPalette(background='#FFFFFF', foreground='black')


data_reset = IntVar()
width = IntVar(value=640)
height = IntVar(value=480)
resolution = StringVar(root)
resolution.set("640x480")  # default value
resolution.trace("w", update_resolution)  # Call update_resolution whenever the value changes

fps = IntVar(value=30)
warmup_frames = StringVar(value="100")
num_captures = StringVar(value="1")

output_dir = StringVar(value='./Capture_Data')
config_file = StringVar(value='./configuration_parameters.json')
output_file = StringVar(value='./point_cloud_combined.ply')
odom_file = StringVar(value='./odometry.log')
mesh_file = StringVar(value='./mesh_combined.ply')
save_individual = IntVar(value=1)

# Configure row and column layout for better alignment
root.grid_columnconfigure(1, weight=1)

# Labels
tk.Button(root, text="Reset Camera Hardware", command=hardware_reset).grid(row=0, column=0, sticky="ew", padx=5, pady=2, columnspan=2)

tk.Label(root, text="Select Directory to Save Captures:").grid(row=1, column=0, sticky="e")
tk.Button(root, text="Select Directory", command=select_output_dir).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
tk.Label(root, textvariable=output_dir).grid(row=2, column=0, columnspan=2, sticky="ew")

tk.Label(root, text="Clear Existing Capture Data?:").grid(row=3, column=0, sticky="e")
tk.Radiobutton(root, text="Yes", variable=data_reset, value=1).grid(row=4, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=data_reset, value=0).grid(row=3, column=1, sticky="w")

tk.Label(root, text="Resolution:").grid(row=5, column=0, sticky="e")
resolutions = ["640x480", "1280x720"]
tk.OptionMenu(root, resolution, *resolutions).grid(row=5, column=1, sticky="ew")


tk.Label(root, text="FPS:").grid(row=6, column=0, sticky="e")
tk.OptionMenu(root, fps, 30, 60).grid(row=6, column=1, sticky="ew")

tk.Label(root, text="Warmup Frames:").grid(row=7, column=0, sticky="e")
tk.Entry(root, textvariable=warmup_frames).grid(row=7, column=1, sticky="ew", padx=5, pady=2)

tk.Label(root, text="Number of Captures:").grid(row=8, column=0, sticky="e")
tk.Entry(root, textvariable=num_captures).grid(row=8, column=1, sticky="ew", padx=5, pady=2)

tk.Label(root, text="Calibration Config File (JSON):").grid(row=9, column=0, sticky="e")
tk.Button(root, text="Select File", command=select_config_file).grid(row=9, column=1, sticky="ew", padx=5, pady=2)
tk.Label(root, textvariable=config_file).grid(row=10, column=0, columnspan=2, sticky="ew")

tk.Label(root, text="Path to Save Point Cloud:").grid(row=11, column=0, sticky="e")
output_entry = tk.Entry(root, textvariable=output_file)
output_entry.grid(row=11, column=1, sticky="ew")
tk.Button(root, text="Browse", command=select_directory_for_output_file).grid(row=11, column=2, padx=5)

tk.Label(root, text="Path to Save Odometry Log:").grid(row=12, column=0, sticky="e")
odom_entry = tk.Entry(root, textvariable=odom_file)
odom_entry.grid(row=12, column=1, sticky="ew")
tk.Button(root, text="Browse", command=select_directory_for_odom_file).grid(row=12, column=2, padx=5)

tk.Label(root, text="Path to Save Mesh:").grid(row=13, column=0, sticky="e")
mesh_entry = tk.Entry(root, textvariable=mesh_file)
mesh_entry.grid(row=13, column=1, sticky="ew")
tk.Button(root, text="Browse", command=select_directory_for_mesh_file).grid(row=13, column=2, padx=5)

tk.Label(root, text="Save Individual PCDs?:").grid(row=14, column=0, sticky="e")
tk.Radiobutton(root, text="Yes", variable=save_individual, value=1).grid(row=14, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=save_individual, value=0).grid(row=15, column=1, sticky="w")

tk.Button(root, text="Submit", command=submit).grid(row=16, column=0,columnspan=2, pady=4,  sticky="ew")

root.mainloop()