import numpy as np
from plyfile import PlyData
from collections import defaultdict
from tqdm import tqdm
import argparse
from vedo import *
from scipy.ndimage import binary_dilation, binary_closing
import datetime

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
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    filled = np.any(voxel_grid > 0, axis=-1)
    x, y, z = np.where(filled)
    c = voxel_grid[x, y, z] / 255.0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c, marker='s', s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Voxel Grid (MAX face color)")
    plt.tight_layout()
    plt.savefig("voxel_visualization.png", dpi=300)
    plt.close()


def interactive_voxel_viewer(voxel_grid, interior_mask):
    filled = np.any(voxel_grid > 0, axis=-1)
    x, y, z = np.where(filled)
    colors = voxel_grid[x, y, z]  # shape: (N, 3)
    coords = np.column_stack((x, y, z))
    # Interior voxels
    interior_x, interior_y, interior_z = np.where(interior_mask)
    interior_colors = voxel_grid[interior_x, interior_y, interior_z]
    interior_coords = np.column_stack((interior_x, interior_y, interior_z))

    # Create Points objects for all voxels and interior voxels
    all_pts = Points(coords, r=5) 
    all_pts.pointcolors = colors 

    interior_pts = Points(interior_coords, r=5)
    interior_pts.pointcolors = interior_colors

    # Create the Plotter
    plt = Plotter(axes=2)
    plt.add(all_pts)  # Start with all voxels visible
    plt.show(interactive=True, title="Interactive Voxel Viewer (Press 'i' to toggle interior view)", viewup="z")

    # State to track which view is active
    showing_all = True

    # Toggle function
    def toggle_visibility(event):
        nonlocal showing_all
        if event.keypress == "i":  # Toggle with the 'i' key
            if showing_all:
                plt.remove(all_pts)
                plt.add(interior_pts)
                plt.render()
                print("[INFO] Showing interior voxels.")
            else:
                plt.remove(interior_pts)
                plt.add(all_pts)
                plt.render()
                print("[INFO] Showing all voxels.")
            showing_all = not showing_all

    # Bind the toggle function to the Plotter
    plt.add_callback("KeyPress", toggle_visibility)

    # Start the interactive viewer
    plt.interactive()

""""
Function to flood fill the inside of the voxel grid with a specified color.
"""
def fill_inside_voxels(voxel_grid, fill_color=(255, 200, 200)):
    filled = np.any(voxel_grid > 0, axis=-1)

    if not is_voxel_watertight(filled):
        # fill holes in the mesh, if any exist (e.g. mario was not water tight)
        filled = binary_closing(filled, structure=np.ones((3, 3, 3)))

    boundary = np.zeros_like(filled, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    outside = ~filled & boundary
    structure = np.ones((3, 3, 3), dtype=bool)

    for _ in range(voxel_grid.shape[0] * 2):
        new_outside = binary_dilation(outside, structure) & ~filled & ~outside
        if not new_outside.any():
            break
        outside |= new_outside

    interior = ~filled & ~outside
    voxel_grid[interior] = fill_color

    return voxel_grid, interior

def is_voxel_watertight(filled):
    # given filled from fill_inside_voxels, output True if watertight and False otherwise
    # NOTE: this is a util function to check if the voxel grid is watertight

    outside = np.zeros_like(filled, dtype=bool)
    outside[0, :, :] = ~filled[0, :, :]
    outside[-1, :, :] = ~filled[-1, :, :]
    outside[:, 0, :] = ~filled[:, 0, :]
    outside[:, -1, :] = ~filled[:, -1, :]
    outside[:, :, 0] = ~filled[:, :, 0]
    outside[:, :, -1] = ~filled[:, :, -1]

    structure = np.ones((3, 3, 3), dtype=bool)

    for _ in range(filled.shape[0] * 2):
        new_outside = binary_dilation(outside, structure) & ~filled & ~outside
        if not new_outside.any():
            break
        outside |= new_outside

    interior = ~filled & ~outside
    return interior.any()
    
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

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelize a PLY triangle mesh using MAX-face coloring")
    parser.add_argument('--input', '-i', required=True, help='Path to input .ply file')
    parser.add_argument('--output', '-o', default='output_voxel.npy', help='Path to save voxel .npy file')
    parser.add_argument('--resolution', '-r', type=int, default=64, help='Voxel grid resolution')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the voxel grid')
    parser.add_argument('--interactive', '-int', action='store_true', help='Enable interactive voxel viewer')

    args = parser.parse_args()

    verts, colors, faces = load_ply(args.input)
    voxel_grid = voxelize_max_color_barycentric(verts, colors, faces, resolution=args.resolution)
    voxel_grid, interior_mask = fill_inside_voxels(voxel_grid, fill_color=(255, 200, 200))

    save_voxel_grid_as_points(voxel_grid, args.output)
    # print(f"[✓] Saved voxel grid to {args.output}")

    if args.visualize:
        visualize_voxels(voxel_grid)
    
    if args.interactive:
        interactive_voxel_viewer(voxel_grid, interior_mask)
        print("[✓] Opened interactive voxel viewer. Close the window to exit.")