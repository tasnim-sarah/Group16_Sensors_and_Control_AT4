"""Quick test for updated red/yellow/blue cubes and same-colour bins."""
from pathlib import Path
import numpy as np
import cv2
import mujoco
import matplotlib.pyplot as plt
from camera_module import CameraSensor, add_wrist_camera, aim_camera_once

HERE = Path(__file__).resolve().parent
spec = mujoco.MjSpec.from_file(str(HERE / "universal_robots_ur3e" / "pick_scene.xml"))
for key in list(spec.keys):
    spec.delete(key)
add_wrist_camera(spec)
model = spec.compile()
data = mujoco.MjData(model)
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
mujoco.mj_forward(model, data)
print("Camera alignment:", aim_camera_once(model, data))

# One reproducible shuffle using the team's ORIGINAL disjoint placement bands.
rng = np.random.default_rng(2026)
bands = [(0.28, 0.42, 0.05, 0.20),
         (0.28, 0.42, -0.08, 0.05),
         (0.28, 0.42, -0.20, -0.08)]
rng.shuffle(bands)
for name, (xmin, xmax, ymin, ymax) in zip(
        ["red_cube_joint", "yellow_cube_joint", "blue_cube_joint"], bands):
    adr = int(model.jnt_qposadr[model.joint(name).id])
    # Free joint: position qpos[adr:adr+3], orientation qpos[adr+3:adr+7].
    data.qpos[adr:adr + 2] = [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, width=640, height=480)
try:
    sensor = CameraSensor(model, renderer)
    rgb, depth = sensor.capture(data)
    detections = sensor.detect(data, rgb, depth)
    marked = rgb.copy()
    for name, measurement in detections.items():
        u, v = measurement["pixel"]
        cv2.drawContours(marked, [measurement["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(marked, (round(u), round(v)), 4, (255, 255, 255), -1)
        print(name, "XYZ world (visible surface):",
              np.round(measurement["position_world_m"], 4))
    missing = set(["Red Cube", "Yellow Cube", "Blue Cube"]) - set(detections)
    print("Missing detections:", sorted(missing) if missing else "none")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(marked)
    axes[0].set_title("Three cubes (bins excluded)")
    axes[0].axis("off")
    im = axes[1].imshow(depth, cmap="viridis", vmin=0.2, vmax=1.5)
    axes[1].set_title("Wrist RGB-D depth (m)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], label="metres")
    plt.tight_layout()
    plt.show()
finally:
    renderer.close()
