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
python test.py
```
 
At the prompt, enter the pick order using commas or spaces, full colour names or
single-letter abbreviations, for example `r,b,y`, `y,r,b`, `b y`, or `yellow red`.
The current scene contains red, blue and yellow cubes. Green (`g` or `green`) is
recognized as an order token but reports unavailable until a green cube and
matching destination are added to the scene configuration.

Each cube must pass its grasp, lift, matching-tray placement, and return checks
before the next requested cube starts. A failed pre-grasp or missed-grasp attempt
gets at most one safe return-to-home and camera-reacquisition retry. Failures
while holding a cube stop the sequence instead of silently advancing.
`CUBE_SEED` reproduces the randomized positions; `ENCODER_NOISE_STD` adjusts the
simulated joint encoder noise (default `0.0005` rad), for example:

```powershell
$env:CUBE_SEED = "11"
$env:ENCODER_NOISE_STD = "0.002"
python test.py
```
Every run appends its success and performance summary to `trial_results.csv`.

**What should happen:**
1. The MuJoCo viewer opens with randomized cube positions and a wrist-mounted RGB-D camera. A second window shows the captured wrist-camera RGB and colour-coded depth frames side by side; on Windows, the application positions both windows next to each other. Camera frames render at 320x240 for responsive monitoring and are enlarged for display.
2. The vision window overlays object labels, colour-coded bounding boxes, and measured depths. It also shows raw and Kalman-filtered target positions, visible/occluded/predicted/lost tracking status, PBVS target and TCP error, current task stage, LQR activity, estimated and true joint positions, and observer uncertainty.
3. The monitoring window reuses the RGB-D frames captured for localization; it does not open a webcam or request extra camera renders. Its display can be closed with `Q` without stopping the robot. It is enabled for every run.
4. PBVS updates short Cartesian waypoints from current detections; numerical IK supplies joint references, and a joint Kalman observer plus discrete LQR drives all six position actuators. If a stationary target becomes occluded, PBVS continues from its last valid filtered position rather than freezing at an intermediate waypoint.
5. The terminal prints state-feedback diagnostics during each move (desired and estimated TCP, estimated and true joint tracking error, IK residual, actuator target, and saturation), and aborts the pick sequence if a required PBVS move fails.
6. Camera updates are paced to at most 30 Hz when processing is faster, without adding a fixed sleep when RGB-D rendering is already the bottleneck. Terminal metrics report camera-pose and joint-state estimation error, position-actuator model residual, per-object RGB-D localization RMSE and large-error counts, rejected out-of-workspace colour detections, pick/place success, and mean capture/detection/display time with simulation real-time factor.
 
## How ground truth is handled
- **Noisy joint encoders → state estimator → forward kinematics.** This gives the TCP and camera poses used for IK and vision localization.
- **Camera only → cube positions** (RGB-D detection and a Kalman filter).
- **MuJoCo ground truth** is used only to generate noisy encoder readings, animate the moving cube, and evaluate tracking, estimation, camera-pose, and placement performance.
## Results
| Test | Attempts | Successful | Success rate |
|---|---|---|---|
| Yellow cube, seed 7, live RGB-D monitor | 1 | 1 | 100% |
| Moving red cube | | | |
| Higher encoder noise | | | |

On the measured live-monitor run, the wrist RGB-D render was 320x240, average camera capture took 59.4 ms/frame, detection 1.9 ms/frame, and display composition/event handling 4.5 ms/frame. The monitor updated at 14.4 FPS and the simulation ran at 0.48x real time. These timings depend on the machine and graphics backend; the terminal reports them again for each run.
 
## Acknowledgements

 