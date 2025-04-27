import numpy as np
from plyfile import PlyData
from collections import defaultdict
from tqdm import tqdm
import argparse
from vedo import *
from scipy.ndimage import binary_dilation, binary_closing
import datetime
import signal
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import plotly.graph_objects as go
import scipy.ndimage


def load_ply(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    face = ply['face']
    
    verts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T.astype(np.uint8)
    faces = np.vstack(face['vertex_indices'])
    
    return verts, colors, faces

def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def voxelize_max_color_barycentric(verts, colors, faces, resolution=64):
    min_bounds = verts.min(axis=0)
    max_bounds = verts.max(axis=0)
    scale = (resolution - 1) / (max_bounds - min_bounds + 1e-8)
    v_scaled = (verts - min_bounds) * scale

    voxel_grid = np.zeros((resolution, resolution, resolution, 3), dtype=np.uint8)
    voxel_area = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    def barycentric_coords(p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return -1, -1, -1  # degenerate triangle
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    for tri_idx in tqdm(range(len(faces)), desc="Voxelizing (barycentric MAX)"):
        idx0, idx1, idx2 = faces[tri_idx]
        v0, v1, v2 = v_scaled[[idx0, idx1, idx2]]
        c0, c1, c2 = colors[[idx0, idx1, idx2]]

        min_corner = np.floor(np.minimum.reduce([v0, v1, v2])).astype(int)
        max_corner = np.ceil(np.maximum.reduce([v0, v1, v2])).astype(int)
        min_corner = np.clip(min_corner, 0, resolution - 1)
        max_corner = np.clip(max_corner, 0, resolution - 1)

        tri_area = triangle_area(v0, v1, v2)

        for x in range(min_corner[0], max_corner[0] + 1):
            for y in range(min_corner[1], max_corner[1] + 1):
                for z in range(min_corner[2], max_corner[2] + 1):
                    p = np.array([x, y, z], dtype=np.float32)  # voxel center
                    u, v, w = barycentric_coords(p, v0, v1, v2)
                    if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
                        interpolated_color = (u * c0 + v * c1 + w * c2).astype(np.uint8)
                        if tri_area > voxel_area[x, y, z]:
                            voxel_grid[x, y, z] = interpolated_color
                            voxel_area[x, y, z] = tri_area

    return voxel_grid

def visualize_voxels(voxel_grid):

    filled = np.any(voxel_grid > 0, axis=-1)  # (X, Y, Z)
    colors = voxel_grid / 255.0  # (X, Y, Z, 3) normalized to [0,1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Only pass color where voxels are filled
    facecolors = np.zeros(filled.shape + (4,), dtype=np.float32)  # RGBA
    facecolors[..., :3] = colors  # RGB
    facecolors[..., 3] = filled  # Alpha = 1 where filled, 0 otherwise

    ax.voxels(
        filled,
        facecolors=facecolors,
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Voxel Grid (MAX face color)")
    plt.tight_layout()
    ax.mouse_init()

    plt.savefig("voxel_visualization.png", dpi=300)
    plt.show()

def interactive_voxel_viewer(voxel_grid):
    filled = np.any(voxel_grid > 0, axis=-1)
    x, y, z = np.where(filled)
    colors = voxel_grid[x, y, z]  # shape: (N, 3)
    coords = np.column_stack((x, y, z))


    # Create Points objects for all voxels and interior voxels
    all_pts = Points(coords, r=5) 
    all_pts.pointcolors = colors 

    # Create the Plotter
    plt = Plotter(axes=2)
    plt.add(all_pts)  # Start with all voxels visible
    plt.show(interactive=True, title="Interactive Voxel Viewer", viewup="z")

    # State to track which view is active
    showing_all = True

    # Start the interactive viewer
    plt.interactive()
    
    plt.close()
    return

# function to just keep the largest connected component inside the voxel grid and remove the rest
def clean_inside(inside_mask, min_size=10):
    structure = np.ones((3,3,3), dtype=np.int32)  # 6-connectivity
    labeled, num_features = scipy.ndimage.label(inside_mask, structure=structure)

    print(f"Found {num_features} inside components.")

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # background

    # Find components larger than min_size
    valid_labels = np.where(component_sizes >= min_size)[0]
    print(f"Keeping {len(valid_labels)} components larger than {min_size} voxels.")

    # Create a mask for all valid components
    cleaned = np.isin(labeled, valid_labels)

    return cleaned

""""
Function to flood fill the inside of the voxel grid with a specified color.
"""
def fill_inside_voxels(voxel_grid, fill_color=(255, 200, 200)):
    print(voxel_grid.shape)
    original_colors = voxel_grid.copy()

    filled = np.any(voxel_grid > 0, axis=-1)
    # dilation -> closing -> dilation
    filled = scipy.ndimage.binary_dilation(filled, iterations=1)
    filled = scipy.ndimage.binary_closing(filled, iterations=1)
    sealed_filled = scipy.ndimage.binary_dilation(filled, iterations=1)


    # create same shape as filled 
    boundary = np.zeros_like(filled, dtype=bool)

    boundary[0, :, :] = True
    boundary[-1, :, :] = True

    boundary[:, 0, :] = True
    boundary[:, -1, :] = True

    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    # create empty array to store the flood fill
    flood_fill = np.zeros_like(filled, dtype=bool)

    # use bfs to flood fill outside the voxel grid starting from a boundary point
    queue = deque()

    # Enqueue all empty boundary voxels
    X, Y, Z = filled.shape
    for x in [0, X-1]:
        for y in range(Y):
            for z in range(Z):
                if not sealed_filled[x, y, z]:
                    queue.append((x, y, z))
    for y in [0, Y-1]:
        for x in range(X):
            for z in range(Z):
                if not sealed_filled[x, y, z]:
                    queue.append((x, y, z))
    for z in [0, Z-1]:
        for x in range(X):
            for y in range(Y):
                if not sealed_filled[x, y, z]:
                    queue.append((x, y, z))

    while queue:
        x, y, z = queue.popleft()

        # Check bounds
        if x < 0 or x >= sealed_filled.shape[0] or y < 0 or y >= sealed_filled.shape[1] or z < 0 or z >= sealed_filled.shape[2]:
            continue

        # If voxel is filled or already flood-filled, skip
        if sealed_filled[x, y, z] or flood_fill[x, y, z]:
            continue

        # Otherwise, mark as flood filled
        flood_fill[x, y, z] = True

        # Add neighbors
        queue.extend([
            (x-1, y, z), (x+1, y, z),
            (x, y-1, z), (x, y+1, z),
            (x, y, z-1), (x, y, z+1)
        ])
        
    print(np.sum(flood_fill))
    print(np.sum(filled))

    # mark voxel grid as the inverse of flood fill
    inside_filled = ~flood_fill & ~sealed_filled & ~filled
    inside_filled = clean_inside(inside_filled, min_size=5)

    print(np.sum(inside_filled))

    # fill the inside of the voxel grid with the fill color
    voxel_grid[inside_filled, :] = fill_color

    return voxel_grid
        

# def is_voxel_watertight(filled):
#     # given filled from fill_inside_voxels, output True if watertight and False otherwise
#     # NOTE: this is a util function to check if the voxel grid is watertight

#     outside = np.zeros_like(filled, dtype=bool)
#     outside[0, :, :] = ~filled[0, :, :]
#     outside[-1, :, :] = ~filled[-1, :, :]
#     outside[:, 0, :] = ~filled[:, 0, :]
#     outside[:, -1, :] = ~filled[:, -1, :]
#     outside[:, :, 0] = ~filled[:, :, 0]
#     outside[:, :, -1] = ~filled[:, :, -1]

#     structure = np.ones((3, 3, 3), dtype=bool)

#     for _ in range(filled.shape[0] * 2):
#         new_outside = binary_dilation(outside, structure) & ~filled & ~outside
#         if not new_outside.any():
#             break
#         outside |= new_outside

#     interior = ~filled & ~outside
#     return interior.any()
    
def save_voxel_grid_as_points(voxel_grid, output_path):
    """
    Save the voxel grid in the format [x, y, z, r, g, b].
    """
    # Find filled voxels
    filled = np.any(voxel_grid > 0, axis=-1)
    x, y, z = np.where(filled)
    colors = voxel_grid[x, y, z]  # shape: (N, 3)

    # Combine coordinates and colors into a single array
    points = np.column_stack((x, y, z, colors))  # shape: (N, 6)

    # Save as .npy file
    np.save(output_path, points)
    print(f"[✓] Saved voxel grid as points to {output_path}")

def clean_exit(*args):
    print("\n[INFO] Exiting gracefully...")
    sys.exit(0)

# ---------- Main ----------
if __name__ == "__main__":
    # Register cleanup signals
    signal.signal(signal.SIGINT, clean_exit)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, clean_exit)  # Handle termination signals

    parser = argparse.ArgumentParser(description="Voxelize a PLY triangle mesh using MAX-face coloring")
    parser.add_argument('--input', '-i', required=True, help='Path to input .ply file')
    parser.add_argument('--output', '-o', default='output_voxel.npy', help='Path to save voxel .npy file')
    parser.add_argument('--resolution', '-r', type=int, default=64, help='Voxel grid resolution')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the voxel grid')
    parser.add_argument('--interactive', '-int', action='store_true', help='Enable interactive voxel viewer')

    args = parser.parse_args()

    verts, colors, faces = load_ply(args.input)
    voxel_grid = voxelize_max_color_barycentric(verts, colors, faces, resolution=args.resolution)
    voxel_grid = fill_inside_voxels(voxel_grid, fill_color=(255, 200, 200))

    save_voxel_grid_as_points(voxel_grid, args.output)
    print(f"[✓] Saved voxel grid to {args.output}")

    if args.visualize:
        visualize_voxels(voxel_grid)
    
    if args.interactive:
        interactive_voxel_viewer(voxel_grid)

        print("[✓] Opened interactive voxel viewer. Close the window to exit.")