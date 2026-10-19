import math
import cv2
import numpy as np
import mujoco
from pathlib import Path

# ---------- Scene (same as before) ----------
spec = mujoco.MjSpec.from_file(
    str(Path(__file__).parent / "universal_robots_ur3e" / "pick_scene.xml"))
for key in list(spec.keys):
    spec.delete(key)

wrist = spec.body("wrist_3_link")
wrist.add_geom(name="camera_shell", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[0.045, 0.0125, 0.0125], pos=[0, 0.08, 0.05],
               rgba=[0.15, 0.15, 0.15, 1], mass=0.072, contype=0, conaffinity=0)
wrist_cam = wrist.add_camera(name="wrist_cam", pos=[0.0325, 0.095, 0.05], fovy=51.28)
wrist_cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
wrist_cam.alt.xyaxes = [1, 0, 0, 0, 0, 1]

model = spec.compile()
data = mujoco.MjData(model)

HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME

rng = np.random.default_rng(2026)
spots = [(0.28, 0.42, 0.05, 0.20), (0.28, 0.42, -0.08, 0.05), (0.28, 0.42, -0.20, -0.08)]
rng.shuffle(spots)
for name, (x0, x1, y0, y1) in zip(
        ["red_cube_joint", "yellow_cube_joint", "blue_cube_joint"], spots):
    adr = model.jnt_qposadr[model.joint(name).id]
    data.qpos[adr:adr + 2] = [rng.uniform(x0, x1), rng.uniform(y0, y1)]
mujoco.mj_forward(model, data)

# ---------- Camera + projection ----------
W, H = 640, 480
cam_id = model.camera("wrist_cam").id
f_px = (H / 2) / math.tan(math.radians(model.cam_fovy[cam_id]) / 2)  # focal length in pixels
renderer = mujoco.Renderer(model, height=H, width=W)

def project(p):
    """World point -> pixel (u, v), or None if behind the camera."""
    R = data.cam_xmat[cam_id].reshape(3, 3)
    pc = R.T @ (p - data.cam_xpos[cam_id])   # point in camera frame (camera looks along -z)
    depth = -pc[2]
    if depth <= 0:
        return None
    return W / 2 + f_px * pc[0] / depth, H / 2 - f_px * pc[1] / depth

watch = ["red_cube", "yellow_cube", "blue_cube", "red_bin", "yellow_bin", "blue_bin"]

# ---------- Sliders: value 0..628 maps to -3.14..+3.14 rad ----------
joint_names = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
win = "Wrist camera"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
for i, n in enumerate(joint_names):
    cv2.createTrackbar(n, win, int(round(HOME[i] * 100)) + 314, 628, lambda v: None)

print("Drag sliders. 'p' = print joint angles, 'q' = quit.")
while True:
    q = np.array([(cv2.getTrackbarPos(n, win) - 314) / 100 for n in joint_names])
    data.ctrl[:6] = q
    for _ in range(16):                      # ~1/30 s of physics per frame
        mujoco.mj_step(model, data)

    renderer.update_scene(data, camera="wrist_cam")
    frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

    y = 25
    for label in watch:
        p = project(data.body(label).xpos)
        visible = p is not None and 0 <= p[0] < W and 0 <= p[1] < H
        color = (0, 220, 0) if visible else (0, 0, 255)
        cv2.putText(frame, f"{label}: {'IN VIEW' if visible else 'out'}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if visible:
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, (255, 255, 255), 2)
        y += 20

    cv2.imshow(win, frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("p"):
        print("HOME =", np.round(q, 4).tolist())
    elif key == ord("q"):
        break

renderer.close()
cv2.destroyAllWindows()