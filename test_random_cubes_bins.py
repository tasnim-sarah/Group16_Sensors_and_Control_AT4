"""RGB-D colour + geometry test for 3 cubes and 3 bins.

Run beside camera_module.py and universal_robots_ur3e/pick_scene.xml:
    python test_random_cubes_bins.py

Standalone test: does not edit either existing file. Detection does not use
MuJoCo object IDs/positions; they are used only to randomise the test scene
and to report independent validation accuracy.
"""
from pathlib import Path
import time
import cv2
import mujoco
import numpy as np
import matplotlib.pyplot as plt

from camera_module import COLOUR_RANGES

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "universal_robots_ur3e" / "pick_scene.xml"
W, H = 640, 480  # Fits the default MuJoCo offscreen framebuffer
LABELS = ("red", "yellow", "blue")


def hsv_mask(hsv, colour):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    # Yellow is often pale in the renderer, so use a more tolerant HSV range.
    ranges = COLOUR_RANGES[f"{colour.capitalize()} Cube"]
    if colour == "yellow":
        ranges = [([18, 65, 85], [43, 255, 255])]
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def place_bin(model, data, name, xy):
    """Move a bin without changing its original height or orientation."""
    bid = model.body(name).id
    jids = [j for j in range(model.njnt) if model.jnt_bodyid[j] == bid]
    if jids:
        for jid in jids:
            kind = model.jnt_type[jid]
            adr = model.jnt_qposadr[jid]
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                data.qpos[adr:adr+2] = xy
                return
            if kind == mujoco.mjtJoint.mjJNT_SLIDE:
                raise RuntimeError(f"{name} uses slide joints; unsupported placement")
        raise RuntimeError(f"{name}: unsupported joint type for repositioning")
    parent = model.body_parentid[bid]
    if parent != 0:
        raise RuntimeError(f"{name} is attached to another body; cannot move in world XY safely")
    model.body_pos[bid, :2] = xy


def place_cube(model, data, name, xy):
    jid = model.joint(name + "_joint").id
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"{name}_joint is not a free joint")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr+2] = xy


def classify_shape(mask, depth, contour):
    """Use *observed RGB-D geometry*, not an object's known identity/location.

    Bin: open centre (a hole in the coloured rim) and/or a centre recessed
    below the colour-matched rim. Cube: relatively filled, flat top.
    Return shape, evidence and measurements for inspection.
    """
    region = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(region, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    center = ((xx-(x+w/2))/(max(w, 1)*0.27))**2 + ((yy-(y+h/2))/(max(h, 1)*0.27))**2 <= 1
    colored = (mask > 0) & (region > 0)
    center_inside = center & (region > 0)
    fill = float(np.count_nonzero(colored) / max(1, np.count_nonzero(region)))
    center_fill = float(np.count_nonzero(colored & center_inside) / max(1, np.count_nonzero(center_inside)))
    good = np.isfinite(depth) & (depth > 0)
    rim_values = depth[colored & good]
    center_values = depth[center_inside & good]
    dip = float(np.median(center_values) - np.percentile(rim_values, 20)) if len(center_values)>8 and len(rim_values)>20 else float('nan')
    # A hollow bin can have coloured floor INSIDE it, so do not require
    # the centre to be dark when its depth recess is clearly measurable.
    hole_evidence = center_fill < 0.65
    depth_evidence = np.isfinite(dip) and dip > 0.017
    # A significantly hollow coloured region is evidence of a receptacle.
    if hole_evidence or depth_evidence:
        shape = "BIN"
    elif center_fill >= 0.88 and (not np.isfinite(dip) or dip < 0.018):
        shape = "CUBE"
    else:
        shape = "UNCERTAIN"
    return shape, {"fill":fill, "centre_fill":center_fill, "depth_recess_m":dip}


def detect(rgb, depth):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Exclude distant patterned floor using *measured depth*, not object IDs.
    # Infer a threshold around the close-depth mode (table + objects).
    finite = depth[np.isfinite(depth) & (depth > 0.05)]
    if finite.size < 100:
        raise ValueError("No valid depth readings from overhead camera")
    near = float(np.percentile(finite, 30))
    far_cutoff = near + 0.23
    workspace = ((depth > 0.05) & np.isfinite(depth) &
                 (depth < far_cutoff)).astype(np.uint8) * 255
    results = []
    for colour in LABELS:
        mask = cv2.bitwise_and(hsv_mask(hsv, colour), workspace)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 35 or area > 2500:
                continue
            shape, features = classify_shape(mask, depth, contour)
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            u, v = moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]
            results.append(dict(colour=colour.upper(), shape=shape, uv=(int(u),int(v)),
                                contour=contour, area=area, **features))
    print(f"Depth workspace cutoff: {far_cutoff:.3f} m (near depth {near:.3f} m)")
    return results


def main():
    if not SCENE.exists():
        raise FileNotFoundError(f"Missing scene file: {SCENE}")
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for key in list(spec.keys):
        spec.delete(key)
    # TEST observation camera, positioned from above to see cubes AND bins.
    # It supplies genuine rendered RGB and metric depth; it is not wrist-mounted.
    spec.worldbody.add_camera(name="sorting_test_overhead", pos=[0.40, 0, 1.2], fovy=63)
    model = spec.compile()
    data = mujoco.MjData(model)
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
    data.qpos[:6] = home
    data.ctrl[:6] = home
    # Six separated positions, randomly assigned to object identities on EVERY run.
    # Equal colours never occupy the same slot. Separation avoids collisions.
    rng = np.random.default_rng()
    slots = [(x, y) for y in (-0.18, 0.18) for x in (0.27, 0.45, 0.63)]
    rng.shuffle(slots)
    truth = {}
    for (colour, kind), (x,y) in zip(
        [(c,k) for c in LABELS for k in ("cube", "bin")], slots
    ):
        xy = np.array([x + rng.uniform(-0.016,0.016), y+rng.uniform(-0.016,0.016)])
        if kind == "cube":
            place_cube(model, data, colour+"_cube", xy)
        else:
            place_bin(model, data, colour+"_bin", xy)
        truth[(colour,kind)] = xy
    mujoco.mj_forward(model, data)
    # Shift framing according to TEST placements only. Classification does not
    # access truth. Camera looks down its -Z axis (identity quaternion).
    camid = model.camera("sorting_test_overhead").id
    model.cam_pos[camid, :3] = [0.45, 0.0, 1.25]
    model.cam_quat[camid] = [1,0,0,0]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=H, width=W)
    try:
        renderer.update_scene(data, camera="sorting_test_overhead")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            renderer.update_scene(data, camera="sorting_test_overhead")
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
    finally:
        renderer.close()
    found = detect(rgb, depth)
    marked = rgb.copy()
    print("\n--- Detected using RGB + depth (NOT object labels) ---")
    for d in found:
        colour = d['colour']; shape = d['shape']; u,v = d['uv']
        print(f"{colour:6s} {shape:9s} pixel=({u:3d},{v:3d}) | colour fill={d['fill']:.2f}, "
              f"centre fill={d['centre_fill']:.2f}, depth recess={d['depth_recess_m']:.3f} m")
        cv2.drawContours(marked, [d['contour']], -1, (30,255,90), 2)
        cv2.putText(marked, f"{colour} {shape}", (max(0,u-45), max(18,v-15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.circle(marked, (u,v), 3, (255,255,255), -1)
    expected = {(c.upper(), k.upper()) for c in LABELS for k in ("cube","bin")}
    actual = {(d["colour"], d["shape"]) for d in found}
    print("\nExpected:", sorted(expected))
    print("Detected:", sorted(actual))
    print("Missing:", sorted(expected-actual) or "none")
    print("Unexpected:", sorted(actual-expected) or "none")
    print("PASS" if actual == expected and len(found)==6 else "CHECK IMAGE / TUNE SHAPE THRESHOLDS")
    fig, axes = plt.subplots(1,2,figsize=(15,6))
    axes[0].imshow(marked); axes[0].set_title("Random positions: RGB + classified shape"); axes[0].axis('off')
    im=axes[1].imshow(depth,cmap='viridis'); axes[1].set_title("Real depth image (metres)"); axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], label="depth (m)")
    fig.tight_layout()
    out = ROOT / "random_cubes_bins_result.png"
    fig.savefig(out, dpi=150)
    print(f"Saved visual result: {out}")
    plt.show()


if __name__ == '__main__':
    main()
