from PyQt5 import QtWidgets, QtCore
from vispy import app, scene
import torch
import numpy as np
import sys
from vispy.color import Color
from vispy.geometry import create_box
from vispy.scene.visuals import Mesh
from vispy.scene import SceneCanvas
from vispy.scene.events import SceneMouseEvent
import signal
from vispy.scene.visuals import Line

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nca_model.NCA import NCA

# ==== Model + Setup ====
model_name = "mario_curriculum3wide_epochs_2400"
ckpt_path = f"../ckpts/{model_name}.pth"
env_dim = 32
n_channels = 16
batch_size = 8
input_channels = 16
learn_seed = True
seed_std = 0.05
update_prob = 0.9 # 0.75
over_to_under_penalty = 1

epochs = 1000
min_iter, max_iter = 48, 64 # 96, 128
learning_rate = 2e-4
weight_decay = 0
alive_thres = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NCA(input_channels, 
            env_dim, 
            learn_seed=learn_seed, 
            update_prob=update_prob, 
            alive_thres=alive_thres, 
            overgrowth_to_undergrowth_penalty=over_to_under_penalty)

model.to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

x = model.seed.unsqueeze(0)
print(x)
living_mask = (x[:,3:4] > model.alive_thres).float()
eval_iter = 0

# ==== PyQt Main Window ====
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NCA Viewer")

        # Central widget: split canvas and buttons
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', parent=central_widget, size=(800, 600))
        layout.addWidget(self.canvas.native)

        # VisPy 3D setup
        self.view = self.canvas.central_widget.add_view()
        center = env_dim // 2

        self.view.camera = scene.cameras.TurntableCamera()
        self.view.camera.center = (center, center, center)
        self.view.camera.set_range(x=(0, env_dim), y=(0, env_dim), z=(0, env_dim))
        self.view.camera.distance = 3 * env_dim
        # self.scatter = scene.visuals.Markers(parent=self.view.scene)
        # self.scatter.antialias = 0
        self.mesh = None
        #self.voxel_group = scene.Node(parent=self.view.scene)

        # add mouse click 
        self.canvas.events.mouse_press.connect(self.on_mouse_press)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(button_layout)

        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.pause_button = QtWidgets.QPushButton("⏸ Pause")
        self.reset_button = QtWidgets.QPushButton("🔄 Reset")
        self.random_damage_button = QtWidgets.QPushButton("💥 Random Damage")
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.random_damage_button)

        self.play_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.stop_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.random_damage_button.clicked.connect(self.random_damage)

        # Timer for auto-update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step_model)
        self.is_running = False

        self.update_visual(torch.clamp(x[0, :4], 0., 1.))


    def update_visual(self, voxels, highlight_voxel = None):
        rgba = voxels.permute(1, 2, 3, 0).detach().cpu().numpy()
        alive = rgba[..., 3] > model.alive_thres
        coords = np.argwhere(alive)
        #print(coords)
        colors = rgba[alive][..., :3]
        colors = np.clip(colors, 0, 1)

        # Highlight a specific voxel if provided
        # if highlight_voxel is not None:
        #     for i, coord in enumerate(coords):
        #         if np.array_equal(coord, highlight_voxel):
        #             colors[i] = [0, 1, 0]  # Set the color to red


        if hasattr(self, "mesh") and self.mesh is not None:
            self.mesh.parent = None

        if len(coords) == 0:
            return

        # Create a unit cube (centered at 0)
        box_vertices, box_faces, _ = create_box(width=1.0, height=1.0, depth=1.0)

        all_vertices = []
        all_faces = []
        all_colors = []

        for i, (pos, color) in enumerate(zip(coords, colors)):
            # Translate cube to voxel location
            translated_vertices = box_vertices['position'] + pos
            all_vertices.append(translated_vertices)

            # Offset faces by vertex count
            all_faces.append(box_faces + i * box_vertices.shape[0])

            # Repeat color for each face vertex
            all_colors.append(np.tile(color, (box_vertices.shape[0], 1)))

        # Stack all cube geometries into one mesh
        V = np.vstack(all_vertices)
        F = np.vstack(all_faces)
        C = np.vstack(all_colors)

        self.mesh = Mesh(vertices=V, faces=F, vertex_colors=C, shading='flat', parent=self.view.scene)

    def get_mouse_ray(self,view, x, y):

        # Make 2 points: near and far in normalized screen coords (z=0 is near, z=1 is far)
        # use homogenous coordinates
        screen_near = np.array([x, y, 0, 1])
        screen_far = np.array([x, y, 1, 1])

        # Map screen to scene with the given imap transform which maps canvas coordinates to scene coordinates
        # p0 is directly under the mouse on the "near" plane
        # p1 is directly under the mouse on the "far" plane
        p0 = view.scene.transform.imap(screen_near)
        p1 = view.scene.transform.imap(screen_far)

        # Normalize with homogenous division
        p0 /= p0[3]
        p1 /= p1[3]

        # origin is the near point, direction is vector from near to far
        origin = p0[:3]
        direction = p1[:3] - origin

        # normalize direction to make it a unit vector
        direction /= np.linalg.norm(direction)
        print(f"Ray origin: {origin}, Ray direction: {direction}")

        return origin, direction
    
    def fuzzy_voxel_hit(self, origin, direction, voxel_grid, voxel_size=1.0, radius=0.75):
        global x, living_mask
        # grab voxel grid shape and alive voxels
        grid_shape = voxel_grid.shape
        alive_voxels = np.argwhere(voxel_grid)  # (N, 3)
        # grab direction
        direction = direction / np.linalg.norm(direction)

        # find the closest voxel to the ray
        best_hit = None
        closest_t = np.inf

        # iterate through all alive voxels
        for voxel in alive_voxels:
            # calculate the center of the voxel
            voxel_center = (voxel + 0.5) * voxel_size
            # find distance from center of voxel to the ray origin
            rel = voxel_center - origin

            # project center-to-origin onto ray to find nearest point
            # direction = how far along the ray gets closest to the voxel center
            t = np.dot(rel, direction)
            # find the closest point on the ray to the given voxel center
            nearest = origin + t * direction

            # find the distance from the voxel center to the calculated nearest point on ray
            dist = np.linalg.norm(voxel_center - nearest)

            # check if its within some radius and in FRONT of the ray
            if dist <= radius and t >= 0:
                # if this is the closest hit so far, save it
                if t < closest_t:
                    closest_t = t
                    best_hit = tuple(voxel)

        x[:, :, best_hit[0]:best_hit[0]+6, best_hit[1]:best_hit[1]+6, best_hit[2]:best_hit[2]+6] = 0
        living_mask[:, :, best_hit[0]:best_hit[0]+6, best_hit[1]:best_hit[1]+6, best_hit[2]:best_hit[2]+6] = 0

        return best_hit

    # helper to draw ray as santiy check
    def draw_ray(self,view, origin, direction, length=1000.0):
        end = origin + direction * length
        ray_line = Line(pos=np.array([origin, end]), color='blue', width=5, parent=view.scene)
        return ray_line

    def on_mouse_press(self, event):
        mouse_x, mouse_y = event.pos
        origin, direction = self.get_mouse_ray(self.view, mouse_x, mouse_y)
        # self.draw_ray(self.view, origin, direction)


        rgba = torch.clamp(x[0, :4], 0., 1.)  # (4, X, Y, Z)
        alpha = rgba[3]
        alive_mask = (alpha > model.alive_thres).cpu().numpy()

        hit_voxel = self.fuzzy_voxel_hit(origin, direction, alive_mask, voxel_size=1.0)

        if hit_voxel:
            print("Hit voxel at:", hit_voxel)

        self.update_visual(rgba, highlight_voxel=hit_voxel)
        
    def step_model(self):
        global x, eval_iter
        with torch.no_grad():
            x = model(x)
            eval_iter += 1
            self.update_visual(torch.clamp(x[0, :4], 0., 1.))
            self.setWindowTitle(f"NCA Viewer - Iteration {eval_iter}")

    def start_simulation(self):
        if not self.is_running:
            self.timer.start(100)  # ms per step
            self.is_running = True

    def stop_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False

    def reset_simulation(self):
        global x, eval_iter
        x = model.seed.unsqueeze(0)
        eval_iter = 0
        self.update_visual(torch.clamp(x[0, :4], 0., 1.))
        self.setWindowTitle("NCA Viewer - Reset")

    def random_damage(self):
        global x, living_mask
        for i in range(np.random.randint(4, 12)):
            damage_x = np.random.randint(0, 26)
            damage_y = np.random.randint(0, 26)
            damage_z = np.random.randint(0, 26)
            x[:, :, damage_x:damage_x+6, damage_y:damage_y+6, damage_z:damage_z+6] = 0
            living_mask[:, :, damage_x:damage_x+6, damage_y:damage_y+6, damage_z:damage_z+6] = 0


# ==== Run the Qt app ====
if __name__ == '__main__':
    def clean_exit(*args):
        print("🧹 Cleaning up...")
        QtWidgets.QApplication.quit()
        sys.exit()

    # Register cleanup signals
    signal.signal(signal.SIGINT, clean_exit)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, clean_exit)  # Handle termination signals

    app.use_app('pyqt5')  # Use Qt backend
    qt_app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    qt_app.exec_()