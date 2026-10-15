"""Live wrist RGB-D sensor demonstration for Group 16.

Run from project root: python camera_live_demo.py
Keys in the Matplotlib window:
  N / Space = new random object arrangement
  A = toggle automatic randomisation every ~2 seconds
  1 = clean sensor   2 = mild RGB/depth noise   3 = stronger noise
  S = save current annotated RGB/depth screenshot
  Q / Escape = quit

The estimator uses RGB and depth images only. MuJoCo object positions are
never accessed for sensing. Ground truth is intentionally NOT displayed here.
The camera's mount orientation is chosen once for the given HOME pose.
"""
from pathlib import Path
import time
import mujoco
import numpy as np
import cv2
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCENE = HERE / 'universal_robots_ur3e' / 'pick_scene.xml'
if not SCENE.exists():
    raise FileNotFoundError(f'Missing team scene: {SCENE}')

WIDTH, HEIGHT = 640, 480
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
WORKSPACE_AIM = np.array([0.35, -0.05, 0.03])
BANDS = [(0.28, 0.42, 0.05, 0.20),
         (0.28, 0.42, -0.08, 0.05),
         (0.28, 0.42, -0.20, -0.08)]
OBJECT_JOINTS = {'Red Cube':'cube_joint',
                 'Blue Sphere':'sphere_joint',
                 'Yellow Pyramid':'pyramid_joint'}
HSV_RANGES = {
    'Red Cube': [([0,130,100],[10,255,255]),([170,130,100],[179,255,255])],
    'Blue Sphere': [([100,180,100],[125,255,255])],
    'Yellow Pyramid': [([22,180,170],[35,255,255])],
}
LABEL_RGB = {'Red Cube': (255, 80, 70), 'Blue Sphere': (65, 145, 255),
             'Yellow Pyramid': (245, 220, 0)}
RNG = np.random.default_rng(2026)
KERNEL = np.ones((3,3),np.uint8)

spec = mujoco.MjSpec.from_file(str(SCENE))
for key in list(spec.keys):
    spec.delete(key)
wrist = spec.body('wrist_3_link')
if wrist is None:
    raise RuntimeError('Cannot find wrist_3_link in scene')
cam_spec = wrist.add_camera(name='wrist_cam', pos=[0.0325,0.095,0.05], fovy=51.28)
cam_spec.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
cam_spec.alt.xyaxes = [1,0,0,0,0,1]
model = spec.compile()
data = mujoco.MjData(model)
data.qpos[:6] = HOME
data.ctrl[:6] = HOME
mujoco.mj_forward(model, data)
cam_id = model.camera('wrist_cam').id
wrist_id = model.body('wrist_3_link').id

# Fix mount orientation at the home pose; it remains wrist-relative afterward.
cam_pos = data.cam_xpos[cam_id].copy()
fwd = WORKSPACE_AIM-cam_pos
fwd /= np.linalg.norm(fwd)
z_cam = -fwd
x_cam = np.cross([0.,0.,1.], z_cam)
x_cam /= np.linalg.norm(x_cam)
y_cam = np.cross(z_cam,x_cam)
R_world_cam = np.column_stack([x_cam,y_cam,z_cam])
R_world_wrist = data.xmat[wrist_id].reshape(3,3)
R_wrist_cam = R_world_wrist.T @ R_world_cam
mujoco.mju_mat2Quat(model.cam_quat[cam_id], R_wrist_cam.flatten())
mujoco.mj_forward(model, data)

fovy = np.deg2rad(model.cam_fovy[cam_id])
fx = fy = HEIGHT/(2*np.tan(fovy/2))
cx,cy = (WIDTH-1)/2, (HEIGHT-1)/2
qaddrs = {}
for label,joint_name in OBJECT_JOINTS.items():
    jid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
    if jid < 0:
        raise RuntimeError(f'Missing joint: {joint_name}')
    qaddrs[label] = int(model.jnt_qposadr[jid])

renderer = mujoco.Renderer(model,width=WIDTH,height=HEIGHT)

def randomise():
    shuffled = RNG.permutation(len(BANDS))
    for label, band_idx in zip(OBJECT_JOINTS,shuffled):
        x0,x1,y0,y1 = BANDS[int(band_idx)]
        addr = qaddrs[label]
        data.qpos[addr:addr+2] = [RNG.uniform(x0,x1),RNG.uniform(y0,y1)]
    mujoco.mj_forward(model,data)

def capture():
    renderer.update_scene(data,camera='wrist_cam')
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data,camera='wrist_cam')
        depth = renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()
    return rgb,depth

def detect(rgb,depth):
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    annotated = rgb.copy()
    R = data.cam_xmat[cam_id].reshape(3,3)
    p = data.cam_xpos[cam_id]
    out = {}
    for name, ranges in HSV_RANGES.items():
        mask = np.zeros((HEIGHT,WIDTH),np.uint8)
        for lo,hi in ranges:
            mask |= cv2.inRange(hsv,np.array(lo,np.uint8),np.array(hi,np.uint8))
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,KERNEL)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,KERNEL)
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours,key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if not 80 <= area <= .10*WIDTH*HEIGHT:
            continue
        m = cv2.moments(contour)
        if m['m00'] <= 0:
            continue
        u,v = m['m10']/m['m00'],m['m01']/m['m00']
        cmask = np.zeros((HEIGHT,WIDTH),np.uint8)
        cv2.drawContours(cmask,[contour],-1,255,-1)
        interior = cv2.erode(cmask,KERNEL,iterations=2)
        if not np.any(interior):
            interior = cmask
        samples = depth[interior>0]
        valid = samples[np.isfinite(samples)&(samples>0)]
        if valid.size == 0:
            continue
        d = float(np.median(valid))
        local = np.array([(u-cx)*d/fx,-(v-cy)*d/fy,-d])
        world = R @ local + p
        out[name] = (world,d)
        cv2.drawContours(annotated,[contour],-1,(0,255,0),2)
        cv2.circle(annotated,(round(u),round(v)),4,(255,255,255),-1)
        labelpos = (min(WIDTH-195,max(4,round(u)+10)),max(22,round(v)-12))
        cv2.putText(annotated,name,labelpos,cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),2)
    return annotated,out

fig,(ax_rgb,ax_depth) = plt.subplots(1,2,figsize=(13,6))
fig.subplots_adjust(bottom=.26,wspace=.30)
fig.canvas.manager.set_window_title('Group 16 - LIVE MuJoCo RGB-D Sensing')
blankrgb = np.zeros((HEIGHT,WIDTH,3),np.uint8)
blankdep = np.zeros((HEIGHT,WIDTH),np.float32)
rgb_artist = ax_rgb.imshow(blankrgb)
dep_artist = ax_depth.imshow(blankdep,cmap='viridis',vmin=.2,vmax=1.5)
fig.colorbar(dep_artist,ax=ax_depth,label='Depth (m)',fraction=.05)
ax_rgb.set_title('Wrist camera: RGB detections')
ax_depth.set_title('Wrist camera: depth')
ax_rgb.axis('off');ax_depth.axis('off')
info_text = fig.text(.04,.19,'',fontsize=10,family='monospace',va='top')
fig.text(.04,.04,'N / SPACE: rearrange  |  A: auto on/off  |  1: clean  2: mild noise  3: stronger noise  |  S: save  |  Q: quit',fontsize=10)
state = {'trial':0,'auto':False,'noise':0,'next_time':time.monotonic()+2,'image':None,'needs_refresh':True}

def refresh():
    clean_rgb,clean_depth = capture()
    std_rgb,std_depth,dropout = [(0,0,0),(6,.002,.01),(18,.010,.03)][state['noise']]
    if std_rgb:
        rgb = np.clip(clean_rgb.astype(float)+RNG.normal(0,std_rgb,clean_rgb.shape),0,255).astype(np.uint8)
    else:
        rgb = clean_rgb
    if std_depth:
        depth = clean_depth + RNG.normal(0,std_depth,clean_depth.shape)
        depth[RNG.random(depth.shape)<dropout] = np.nan
    else:
        depth = clean_depth
    overlay,detections = detect(rgb,depth)
    rgb_artist.set_data(overlay)
    dep_artist.set_data(depth)
    state['image'] = overlay
    state['depth_image'] = depth
    lines = [f"Arrangement {state['trial']}    Sensor noise: {['clean','mild','strong'][state['noise']]}    Auto: {'ON' if state['auto'] else 'OFF'}"]
    for name in OBJECT_JOINTS:
        if name in detections:
            xyz,d = detections[name]
            lines.append(f'{name:<16} XYZ world (m): [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]   depth={d:.3f} m')
        else:
            lines.append(f'{name:<16} NOT DETECTED')
    info_text.set_text('\n'.join(lines))
    fig.canvas.draw_idle()
    print('\n'.join(lines),flush=True)

def new_arrangement():
    randomise()
    state['trial']+=1
    state['needs_refresh']=True

def on_key(event):
    key=(event.key or '').lower()
    if key in ('n',' '):
        new_arrangement()
        state['next_time']=time.monotonic()+2
    elif key=='a':
        state['auto']=not state['auto']
        state['next_time']=time.monotonic()+2
        state['needs_refresh']=True
    elif key in ('1','2','3'):
        state['noise']=int(key)-1
        state['needs_refresh']=True
    elif key=='s':
        path=HERE/f'camera_demo_arrangement_{state["trial"]}.png'
        fig.savefig(path,dpi=150)
        print(f'Saved demonstration screenshot: {path}')
    elif key in ('q','escape'):
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event',on_key)
new_arrangement()
try:
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        if state['auto'] and time.monotonic()>=state['next_time']:
            new_arrangement()
            state['next_time']=time.monotonic()+2
        if state['needs_refresh']:
            refresh()
            state['needs_refresh']=False
        plt.pause(.08)
finally:
    renderer.close()
