from PyQt5 import QtWidgets, QtCore
from vispy import app, scene
import torch
import numpy as np
import sys
from vispy.color import Color

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.NCA import NCA

# ==== Model + Setup ====
model_name = "mario_epochs_1000"
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
        self.view.camera = scene.cameras.TurntableCamera(fov=60)
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.antialias = 0
        #self.voxel_group = scene.Node(parent=self.view.scene)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(button_layout)

        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.pause_button = QtWidgets.QPushButton("⏸ Pause")
        self.reset_button = QtWidgets.QPushButton("🔄 Reset")
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_button)

        self.play_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.stop_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)

        # Timer for auto-update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step_model)
        self.is_running = False

        self.update_visual(torch.clamp(x[0, :4], 0., 1.))


    def update_visual(self, voxels):
        rgba = voxels.permute(1, 2, 3, 0).detach().cpu().numpy()
        alive = rgba[..., 3] > model.alive_thres
        coords = np.argwhere(alive)
        colors = rgba[alive][..., :3]
        colors = np.clip(colors, 0, 1)
        if len(coords) > 0:
            self.scatter.set_data(coords, face_color=colors, size=10)
        else:
            self.scatter.set_data(np.zeros((0, 3)), face_color='white', size=10)
        # Clear previous cubes
        # for child in list(self.voxel_group.children):
        #     child.parent = None  # Removes it from the scene

        # rgba = voxels.permute(1, 2, 3, 0).detach().cpu().numpy()
        # alive = rgba[..., 3] > model.alive_thres
        # coords = np.argwhere(alive)
        # colors = rgba[alive][..., :3]
        # colors = np.clip(colors, 0, 1)

        # box_size = 0.95  # Adjust for spacing between cubes

        # for pos, col in zip(coords, colors):
        #     cube = scene.visuals.Box(
        #         width=box_size, height=box_size, depth=box_size,
        #         color=Color(col), edge_color=None, parent=self.voxel_group
        #     )
        #     cube.transform = scene.transforms.STTransform(translate=pos)

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

# ==== Run the Qt app ====
if __name__ == '__main__':
    app.use_app('pyqt5')  # Use Qt backend
    qt_app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    qt_app.exec_()