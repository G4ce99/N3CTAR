from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QWidget, QMainWindow, QHBoxLayout

import os
import sys
import torch
import signal
import numpy as np
from vispy import app, scene
from vispy.color import Color
from vispy.geometry import create_box
from vispy.scene.visuals import Mesh, Line
from vispy.scene import SceneCanvas
from vispy.scene.events import SceneMouseEvent
from vispy.scene.cameras import TurntableCamera
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from vispy.app import use_app
use_app('pyqt5')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nca_model.NCA_hidden_LayerNorm import NCA

# ==== Constants ====
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

# ==== Model + Setup ====
model_name = "final_mario_focused_lin_damage5_p06_1-1_6000"
ckpt_path = f"../ckpts/{model_name}.pth"

model = NCA(input_channels, 
            env_dim, 
            learn_seed=learn_seed, 
            update_prob=update_prob, 
            alive_thres=alive_thres, 
            overgrowth_to_undergrowth_penalty=over_to_under_penalty)

model.to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
model.eval()

x = model.seed.unsqueeze(0)
print(x)
living_mask = (x[:,3:4] > model.alive_thres).float()
eval_iter = 0

#==== PyQt Setup ====

# create a model switcher class for a dropdown menu
class ModelSwitcher(QWidget):
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.dropdown = QComboBox(self)
        self.dropdown.addItems(["Mario 32x32x32", "Duck 32x32x32", "Bald Mario"])
        self.dropdown.currentIndexChanged.connect(self.on_model_change)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.dropdown)
        self.setLayout(self.layout)

    def load_model(self, model_name, ckpt_path):
        # load the model from the model name and checkpoint path 
        model = NCA(input_channels, 
                    env_dim, 
                    learn_seed=learn_seed, 
                    update_prob=update_prob, 
                    alive_thres=alive_thres, 
                    overgrowth_to_undergrowth_penalty=over_to_under_penalty)
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()

        x = model.seed.unsqueeze(0)
        living_mask = (x[:,3:4] > model.alive_thres).float()
        eval_iter = 0

        return model, x, living_mask, eval_iter

    def on_model_change(self, index):
        # 32x32x32 mario
        if index == 0:
            model_name = 'final_mario_focused_lin_damage5_p06_1-1_6000'
            ckpt_path = f"../ckpts/{model_name}.pth"

            self.visualizer.model, self.visualizer.x, self.visualizer.living_mask, self.visualizer.eval_iter = self.load_model(model_name, ckpt_path)
            # reset the model
            self.visualizer.reset_simulation()
            self.visualizer.update_visual(torch.clamp(self.visualizer.x[0, :4], 0., 1.))
            self.visualizer.setWindowTitle("NCA Viewer - Mario 32x32x32")
        # 32x32x32 duck
        elif index == 1:
            model_name = 'final_duck5_lin_damage5_up1_dp04_1-1_4000'
            ckpt_path = f"../ckpts/{model_name}.pth"

            self.visualizer.model, self.visualizer.x, self.visualizer.living_mask, self.visualizer.eval_iter = self.load_model(model_name, ckpt_path)
            # reset the model
            self.visualizer.reset_simulation()
            self.visualizer.update_visual(torch.clamp(self.visualizer.x[0, :4], 0., 1.))
            self.visualizer.setWindowTitle("NCA Viewer - Duck 32x32x32")

        # 32x32x32 bald mario
        elif index == 2:
            model_name = 'mario_curriculum4_ln_damage_4-12_over10'
            ckpt_path = f"../ckpts/{model_name}.pth"

            self.visualizer.model, self.visualizer.x, self.visualizer.living_mask, self.visualizer.eval_iter = self.load_model(model_name, ckpt_path)
            # reset the model
            self.visualizer.reset_simulation()
            self.visualizer.update_visual(torch.clamp(self.visualizer.x[0, :4], 0., 1.))
            self.visualizer.setWindowTitle("NCA Viewer - Bald Mario")


# ==== PyQt Main Window ====
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, model, x, living_mask, eval_iter):
        super().__init__()
        self.setWindowTitle("NCA Viewer")

        # set global variables 
        self.model = model
        self.x = x
        self.living_mask = living_mask
        self.eval_iter = eval_iter

        # Central widget: split canvas and buttons
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='lightgray', parent=central_widget, size=(800, 600))
        layout.addWidget(self.canvas.native)
        self.model_switcher = ModelSwitcher(self)

        # VisPy 3D setup
        self.view = self.canvas.central_widget.add_view()
        center = env_dim // 2

        # camera setup 
        self.view.camera = scene.cameras.TurntableCamera()
        self.view.camera.center = (center, center, center)
        self.view.camera.set_range(x=(0, env_dim), y=(0, env_dim), z=(0, env_dim))
        self.view.camera.distance = 3 * env_dim
        self.mesh = None

        # add mouse click events
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.key_release.connect(self.on_key_release)


        # Button layout to be horizontal bar on the bottom
        button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(button_layout)
        
        # Buttons
        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.pause_button = QtWidgets.QPushButton("⏸ Pause")
        self.reset_button = QtWidgets.QPushButton("🔄 Reset")
        self.random_damage_button = QtWidgets.QPushButton("💥 Random Damage")
        button_layout.addWidget(self.model_switcher)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.random_damage_button)

        # Button actions
        self.play_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.stop_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.random_damage_button.clicked.connect(self.random_damage)

        # Timer for auto-update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step_model)

        # set flags
        self.is_running = False
        self.is_dragging = False
        self.d_key_pressed = False

        self.update_visual(torch.clamp(self.x[0, :4], 0., 1.))


    def update_visual(self, voxels, highlight_voxel = None):
        rgba = voxels.permute(1, 2, 3, 0).detach().cpu().numpy()
        alive = rgba[..., 3] > self.model.alive_thres
        coords = np.argwhere(alive)
        colors = rgba[alive][..., :3]
        colors = np.clip(colors, 0, 1)

        # Reduce saturation
        hsv_colors = rgb_to_hsv(colors)  # Convert RGB to HSV
        hsv_colors[:, 1] *= 0.75  # Reduce saturation
        hsv_colors[:, 1] = np.clip(hsv_colors[:, 1], 0, 1)  # Ensure valid range
        colors = hsv_to_rgb(hsv_colors)  # Convert back to RGB


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

        self.mesh = Mesh(vertices=V, faces=F, vertex_colors=C, shading=None, parent=self.view.scene)

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

        self.x[:, :, best_hit[0]:best_hit[0]+6, best_hit[1]:best_hit[1]+6, best_hit[2]:best_hit[2]+6] = 0
        self.living_mask[:, :, best_hit[0]:best_hit[0]+6, best_hit[1]:best_hit[1]+6, best_hit[2]:best_hit[2]+6] = 0

        return best_hit

    # helper to draw ray as santiy check
    def draw_ray(self,view, origin, direction, length=1000.0):
        end = origin + direction * length
        ray_line = Line(pos=np.array([origin, end]), color='blue', width=5, parent=view.scene)
        return ray_line
    
    # helper to check handle ray intersection
    def handle_ray_hit(self, event):
        mouse_x, mouse_y = event.pos
        origin, direction = self.get_mouse_ray(self.view, mouse_x, mouse_y)
        # self.draw_ray(self.view, origin, direction)


        rgba = torch.clamp(self.x[0, :4], 0., 1.)  # (4, X, Y, Z)
        alpha = rgba[3]
        alive_mask = (alpha > self.model.alive_thres).cpu().numpy()

        hit_voxel = self.fuzzy_voxel_hit(origin, direction, alive_mask, voxel_size=1.0)

        if hit_voxel:
            print("Hit voxel at:", hit_voxel)

        self.update_visual(rgba, highlight_voxel=hit_voxel)

    # check for destruction key press
    def on_key_press(self, event):
        if event.key == 'D':
            self.d_key_pressed = True

    # check for destruction key release
    def on_key_release(self, event):
        if event.key == 'D':
            self.d_key_pressed = False

    # check for mouse press
    def on_mouse_press(self, event):
        # if key is d, start checking for ray intersections
        if self.d_key_pressed:
            self.view.camera.interactive = False
            self.is_dragging = True
            self.handle_ray_hit(event)

    # handle click and drag with destruction
    def on_mouse_move(self, event):
        if self.is_dragging and self.d_key_pressed:
            self.handle_ray_hit(event)
              
    # handle mouse release with dragging for destruction
    def on_mouse_release(self, event):
        if self.is_dragging and self.d_key_pressed:
            self.is_dragging = False
            self.d_key_pressed = False
        self.view.camera.interactive = True
            
    # handle model steps 
    def step_model(self):
        try:
            with torch.no_grad():
                self.x, self.living_mask = self.model(self.x, self.living_mask)
                self.eval_iter += 1
                self.update_visual(torch.clamp(self.x[0, :4], 0., 1.))
                self.setWindowTitle(f"NCA Viewer - Iteration {self.eval_iter}")
        except Exception as e:
            print(f"⚠️ Error in step_model: {e}")
            self.stop_simulation()  # Stop the simulation to prevent further errors
            clean_exit()  # Call the cleanup function

    # start the simulation
    def start_simulation(self):
        if not self.is_running:
            self.timer.start(100)  # ms per step
            self.is_running = True

    # stop the simulation
    def stop_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False

    # 
    def reset_simulation(self):
        self.x = self.model.get_seed().unsqueeze(0)
        self.living_mask = (self.x[:,3:4] > self.model.alive_thres).float()

        self.eval_iter = 0
        self.update_visual(torch.clamp(self.x[0, :4], 0., 1.))
        self.setWindowTitle("NCA Viewer - Reset")

    def random_damage(self):
        for i in range(np.random.randint(4, 12)):
            damage_x = np.random.randint(0, 26)
            damage_y = np.random.randint(0, 26)
            damage_z = np.random.randint(0, 26)
            self.x[:, :, damage_x:damage_x+6, damage_y:damage_y+6, damage_z:damage_z+6] = 0
            self.living_mask[:, :, damage_x:damage_x+6, damage_y:damage_y+6, damage_z:damage_z+6] = 0


# ==== Run the Qt app ===== 
if __name__ == '__main__':
    def clean_exit(*args):
        print("🧹 Cleaning up...")
        QtWidgets.QApplication.quit()
        sys.exit()

    # Register cleanup signals
    signal.signal(signal.SIGINT, clean_exit)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, clean_exit)  # Handle termination signals

    try:
        app.use_app('pyqt5')  # Use Qt backend
        qt_app = QtWidgets.QApplication(sys.argv)
        print(eval)
        window = MainWindow(model, x, living_mask, eval_iter)
        window.show()
        qt_app.exec_()
    except Exception as e:
        print(f"⚠️ Unhandled exception: {e}")
    finally:
        clean_exit()