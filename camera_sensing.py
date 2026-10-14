# ============================================================
# INTEGRATED PROJECT - RGB-D CAMERA SENSING
# Stage 1: RGB and depth capture
# Stage 2: HSV colour-based object detection
# Stage 3: Filtered depth and 3D object localisation
# ============================================================

from pathlib import Path
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ------------------------------------------------------------
# 1. LOAD THE EXISTING MUJOCO SCENE
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent
scene_path = HERE / "universal_robots_ur3e" / "pick_scene.xml"

if not scene_path.exists():
    raise FileNotFoundError(f"Scene not found: {scene_path}\nPlace this script in your project root, beside test_pick_scene.py.")

spec = mujoco.MjSpec.from_file(str(scene_path))
for key in list(spec.keys):
    spec.delete(key)

# ------------------------------------------------------------
# 2. ATTACH THE SAME WRIST CAMERA AS THE TEAM'S SCENE TEST
# ------------------------------------------------------------
wrist = spec.body("wrist_3_link")
if wrist is None:
    raise ValueError("wrist_3_link not found in the scene")

camera = wrist.add_camera(
    name="wrist_cam",
    pos=[0.0325, 0.095, 0.05],
    fovy=51.28,
)
camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
camera.alt.xyaxes = [1, 0, 0, 0, 0, 1]

# ------------------------------------------------------------
# 3. INITIALISE THE ROBOT
# ------------------------------------------------------------
model = spec.compile()
data = mujoco.MjData(model)

HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
mujoco.mj_forward(model, data)

# ------------------------------------------------------------
# 4. AIM WRIST CAMERA AT THE WORKSPACE AT THE HOME POSE
# This sets a fixed mounting orientation relative to the wrist.
# It does NOT track the target when the robot moves.
# ------------------------------------------------------------
cam_id = model.camera("wrist_cam").id
wrist_id = model.body("wrist_3_link").id
cam_pos = data.cam_xpos[cam_id].copy()
target_world = np.array([0.35, -0.05, 0.03])

forward = target_world - cam_pos
forward /= np.linalg.norm(forward)
z_axis = -forward  # MuJoCo optical direction is camera-local -Z
reference = np.array([0.0, 0.0, 1.0])
x_axis = np.cross(reference, z_axis)
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
R_world_camera = np.column_stack((x_axis, y_axis, z_axis))
R_world_wrist = data.xmat[wrist_id].reshape(3, 3)
R_wrist_camera = R_world_wrist.T @ R_world_camera
mujoco.mju_mat2Quat(model.cam_quat[cam_id], R_wrist_camera.flatten())
mujoco.mj_forward(model, data)

R_new = data.cam_xmat[cam_id].reshape(3, 3)
new_forward = -R_new[:, 2]
print("New camera direction:", np.round(new_forward, 3))
print("Target direction:", np.round(forward, 3))
print("New alignment:", np.dot(new_forward, forward))

# ------------------------------------------------------------
# 5. CAPTURE RGB AND DEPTH IMAGES
# ------------------------------------------------------------
WIDTH, HEIGHT = 640, 480
renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
try:
    renderer.update_scene(data, camera="wrist_cam")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_cam")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
finally:
    renderer.close()

print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)
print("Depth range:", np.nanmin(depth), np.nanmax(depth))

# ------------------------------------------------------------
# 6-7. RGB TO HSV AND OBJECT COLOUR THRESHOLDS
# Thresholds tuned for the team's current rendered scene.
# ------------------------------------------------------------
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
colour_ranges = {
    "Red Cube": [([0, 130, 100], [10, 255, 255]),
                 ([170, 130, 100], [179, 255, 255])],
    "Blue Sphere": [([100, 180, 100], [125, 255, 255])],
    "Yellow Pyramid": [([22, 180, 170], [35, 255, 255])],
}

# ------------------------------------------------------------
# 8. DETECT OBJECTS AND SAVE EACH OBJECT'S MASK
# ------------------------------------------------------------
annotated_rgb = rgb.copy()
detected_objects = {}
kernel = np.ones((3, 3), dtype=np.uint8)

for name, ranges in colour_ranges.items():
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        colour_mask = cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
        mask = cv2.bitwise_or(mask, colour_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"{name}: NOT DETECTED")
        continue

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 80 or area > WIDTH * HEIGHT * 0.10:
        print(f"{name}: rejected contour area {area:.0f} px")
        continue

    M = cv2.moments(contour)
    if M["m00"] == 0:
        print(f"{name}: invalid contour")
        continue

    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    object_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(object_mask, [contour], -1, 255, thickness=-1)

    detected_objects[name] = {"pixel": (u, v), "area": area, "mask": object_mask}
    cv2.drawContours(annotated_rgb, [contour], -1, (0, 255, 0), 2)
    cv2.circle(annotated_rgb, (u, v), 5, (255, 255, 255), -1)
    cv2.putText(annotated_rgb, name, (u + 8, max(v - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    print(f"{name}: centre=({u}, {v}), area={area:.0f} px")

# ------------------------------------------------------------
# 9. CAMERA INTRINSICS: PINHOLE APPROXIMATION
# MuJoCo specifies vertical FOV. With square pixels, fx = fy.
# ------------------------------------------------------------
fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
fy = HEIGHT / (2.0 * np.tan(fovy_rad / 2.0))
fx = fy
cx = (WIDTH - 1) / 2.0
cy = (HEIGHT - 1) / 2.0
K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
print("\n===== CAMERA INTRINSICS (FROM FOV) =====")
print(np.round(K, 3))

# Camera-local +X right, +Y up, and -Z forward.
def pixel_to_camera(u, v, depth_m):
    return np.array([
        (u - cx) * depth_m / fx,
        -(v - cy) * depth_m / fy,
        -depth_m,
    ])

# Camera pose is read from MuJoCo, not guessed.
def camera_to_world(point_camera):
    camera_position = data.cam_xpos[cam_id].copy()
    camera_rotation = data.cam_xmat[cam_id].reshape(3, 3).copy()
    return camera_rotation @ point_camera + camera_position

# ------------------------------------------------------------
# 10. FILTER OBJECT DEPTH AND COMPUTE 3D SURFACE POINTS
# Depth is taken from eroded object interior, not background.
# Each XYZ is the estimated visible surface point on the
# centroid ray; it is not necessarily the physical object centre.
# ------------------------------------------------------------
print("\n===== 3D OBJECT LOCALISATION =====")
for name, info in detected_objects.items():
    u, v = info["pixel"]
    interior_mask = cv2.erode(info["mask"], kernel, iterations=2)
    if np.count_nonzero(interior_mask) == 0:
        interior_mask = info["mask"]

    samples = depth[interior_mask > 0]
    valid = samples[np.isfinite(samples) & (samples > 0)]
    if len(valid) == 0:
        print(f"{name}: no valid depth samples")
        continue

    raw_depth = float(depth[v, u])
    median_depth = float(np.median(valid))
    point_camera = pixel_to_camera(u, v, median_depth)
    point_world = camera_to_world(point_camera)
    info.update({
        "raw_depth": raw_depth,
        "median_depth": median_depth,
        "point_camera": point_camera,
        "point_world": point_world,
        "depth_samples": int(len(valid)),
    })

    print(f"\n{name}")
    print(f"  Pixel centre: ({u}, {v})")
    print(f"  Valid depth pixels: {len(valid)}")
    print(f"  Raw centre depth: {raw_depth:.4f} m")
    print(f"  Filtered median depth: {median_depth:.4f} m")
    print("  Camera XYZ (m):", np.round(point_camera, 4))
    print("  World XYZ (m):", np.round(point_world, 4))

# ------------------------------------------------------------
# 11. DISPLAY THE ANNOTATED RGB AND DEPTH IMAGES
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(annotated_rgb)
axes[0].set_title("RGB Camera - Object Detection")
axes[0].axis("off")
depth_img = axes[1].imshow(depth, cmap="viridis")
axes[1].set_title("Depth Camera")
axes[1].axis("off")
fig.colorbar(depth_img, ax=axes[1], label="Depth (m)")
plt.tight_layout()
plt.show()

print("\n========== CAMERA DETECTION SUMMARY ==========")
print("Number of detected objects:", len(detected_objects))
for name, info in detected_objects.items():
    if "point_world" in info:
        print(f"{name}: pixel={info['pixel']}, world surface XYZ={np.round(info['point_world'], 4)} m")
    else:
        print(f"{name}: pixel={info['pixel']}, no valid 3D position")
print("==============================================")

# ============================================================
# STAGE 5: FAIR DEPTH-NOISE AND FILTERING EVALUATION
# ============================================================
# Each method is compared against its OWN noise-free measurement.
# This isolates noise sensitivity from the natural depth difference
# between an object's centre pixel and its visible surface median.

rng = np.random.default_rng(42)  # repeatable experimental results
NOISE_LEVELS = [0.0, 0.002, 0.005, 0.010]  # metres: 0, 2, 5, 10 mm
N_TRIALS = 100
DROPOUT_RATE = 0.02
results = []

print("\n===== DEPTH NOISE ROBUSTNESS TEST (FAIR BASELINES) =====")

for noise_std in NOISE_LEVELS:
    # Zero-noise baseline has no dropout either.
    dropout_rate = 0.0 if noise_std == 0.0 else DROPOUT_RATE

    for name, info in detected_objects.items():
        u, v = info["pixel"]
        object_mask = info["mask"] > 0
        clean_values = depth[object_mask]
        clean_values = clean_values[np.isfinite(clean_values) & (clean_values > 0)]
        clean_raw_depth = float(depth[v, u])

        if clean_values.size == 0 or not np.isfinite(clean_raw_depth) or clean_raw_depth <= 0:
            print(f"{name}: no valid clean reference; skipped")
            continue

        clean_median_depth = float(np.median(clean_values))
        raw_errors = []
        median_errors = []
        raw_failures = 0
        median_failures = 0

        for _ in range(N_TRIALS):
            # Independent Gaussian noise and invalid-pixel dropout.
            noisy_depth = depth + rng.normal(0.0, noise_std, size=depth.shape)
            dropout = rng.random(depth.shape) < dropout_rate
            noisy_depth[dropout] = np.nan

            # Method A: single measured pixel relative to clean pixel.
            raw_depth = noisy_depth[v, u]
            if np.isfinite(raw_depth) and raw_depth > 0:
                raw_errors.append(float(raw_depth - clean_raw_depth))
            else:
                raw_failures += 1

            # Method B: segmented median relative to clean median.
            values = noisy_depth[object_mask]
            valid = values[np.isfinite(values) & (values > 0)]
            if valid.size:
                median_errors.append(float(np.median(valid) - clean_median_depth))
            else:
                median_failures += 1

        raw_rmse = float(np.sqrt(np.mean(np.square(raw_errors))) * 1000) if raw_errors else np.nan
        median_rmse = float(np.sqrt(np.mean(np.square(median_errors))) * 1000) if median_errors else np.nan

        results.append({
            "object": name,
            "noise_mm": noise_std * 1000,
            "raw_rmse_mm": raw_rmse,
            "median_rmse_mm": median_rmse,
            "raw_failures": raw_failures,
            "median_failures": median_failures,
        })
        print(
            f"{name:16s} | Noise: {noise_std * 1000:4.0f} mm | "
            f"Raw RMSE: {raw_rmse:7.2f} mm | "
            f"Median RMSE: {median_rmse:7.2f} mm | "
            f"Failures raw/median: {raw_failures}/{median_failures}"
        )

# ============================================================
# STAGE 5B: SAVE ROBUSTNESS COMPARISON
# ============================================================
fig, axes = plt.subplots(1, len(detected_objects), figsize=(5 * len(detected_objects), 4), squeeze=False)
for ax, name in zip(axes.flat, detected_objects):
    subset = [r for r in results if r["object"] == name]
    ax.plot([r["noise_mm"] for r in subset], [r["raw_rmse_mm"] for r in subset], "o-", label="Single pixel")
    ax.plot([r["noise_mm"] for r in subset], [r["median_rmse_mm"] for r in subset], "s-", label="Object-mask median")
    ax.set_title(name)
    ax.set_xlabel("Added Gaussian depth noise SD (mm)")
    ax.set_ylabel("Depth change RMSE (mm)")
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plot_path = HERE / "depth_robustness_corrected.png"
plt.savefig(plot_path, dpi=200)
plt.show()
print(f"\nCorrected robustness plot saved: {plot_path}")
