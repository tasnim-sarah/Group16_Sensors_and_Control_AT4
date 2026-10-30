# Vision-Guided Pick and Place with a UR3e in MuJoCo
 
A simulated UR3e robot with a wrist-mounted RGB-D camera finds red, yellow and blue cubes on a table, picks each one up and drops it in the bin of the same colour. The robot only uses **estimated** state from noisy sensors: joint encoders go through a state estimator, and cube positions come from the camera and a Kalman filter. Simulator ground truth is used **only to evaluate** the results.
 
41014 Sensors and Control – AT4, Track B (UR3 vision-guided pick in MuJoCo)
 
▶ **Video:** [link](PASTE_LINK_HERE)
 
## Group information
**Group 16**
 
| Member |
|---|
| Tasnim Sarah |
| Maryam Qutub | 
| Princess Kisakye |
| Vanessa Kurniawidi| 
 
## Install info
Tested with Python 3.12 on Windows 11. From a fresh environment:
 
```bash
git clone https://github.com/USERNAME/Group16_Sensors_and_Control_AT4.git
cd Group16_Sensors_and_Control_AT4
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
 
Dependencies (in `requirements.txt`): `mujoco`, `numpy`, `opencv-python`
 
## Run commands
```bash
python pick_place_estimated.py r b y
```
 
`r b y` is the pick order (red, blue, yellow). Any order works, e.g. `y r`.
 
**What should happen:**
1. A window shows the camera's detections at the start. Press any key to continue.
2. The MuJoCo viewer and a live wrist-camera window open.
3. The robot picks the static cubes from their camera positions. For the moving red cube, it watches the cube, learns when it pauses, and grabs it during a pause.
4. Each cube goes into its matching bin. The terminal prints `PLACEMENT CHECK: PASS` per cube and a summary at the end.
Press `Q` in the camera window to stop.
 
## How ground truth is handled
- **Joint encoders + noise → state estimator → forward kinematics.** This gives the tool pose, Jacobian and camera pose used by the controller.
- **Camera only → cube positions** (RGB-D detection and a Kalman filter).
- **Finger position sensors → grasp detection** (not simulator contacts).
- Ground truth appears only in code marked `EVALUATION ONLY`, which prints results and never changes what the robot does.
## Results
| Test | Attempts | Successful | Success rate |
|---|---|---|---|
| Static cubes | | | |
| Moving red cube | | | |
| Higher encoder noise | | | |
 
## Acknowledgements

 