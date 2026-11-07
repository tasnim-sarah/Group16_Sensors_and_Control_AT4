# Vision-Guided Pick and Place with a UR3e in MuJoCo

41014 Sensors and Control, Assessment 4, Track B. Group 16. Simulation only.

A simulated UR3e uses a wrist-mounted RGB-D camera to find red, blue and yellow cubes and place them in matching-colour trays in the requested order. Kalman filters estimate object motion and robot joint states. Position-based visual servoing and inverse kinematics generate references tracked by a discrete LQR controller. The red cube moves on a physical platform and is tracked during approach, descent and gripper closure.

**Final application:** [`Maryam_Final_Integration/main_final_integration.py`](Maryam_Final_Integration/main_final_integration.py). Run this entry point; the Python files alongside it are required support modules. Other scripts in the repository include earlier development versions.

**Project website:** [link](PASTE_WEBSITE_LINK_HERE)

**Demo video:** [Watch the demonstration](https://youtu.be/spyB0FesHyg)

**Technical documentation:** [pipeline walkthrough](Maryam_Final_Integration/docs/pipeline_walkthrough.md) and [recorded validation report](Maryam_Final_Integration/docs/index.html).

## Contributions

### Tasnim Sarah

Tasnim created the repository and supported team communication and organisation. She helped integrate different parts of the simulation and developed project-page content, including theoretical explanations and graphs.

### Maryam Qutub

Maryam initially worked on RGB-D sensing, including wrist-camera colour detection and depth-based localisation of the cubes. She then connected camera measurements and Kalman estimates to position-based visual servoing (PBVS), inverse kinematics and LQR joint control using estimated feedback. She also worked on randomised sorting, live camera monitoring, performance measurements and debugging pickup and transfer failures.

### Princess Kisakye

Princess developed the codespace to support the project requirements and coordinated team roles and responsibilities. She worked on Kalman filtering and further integration, helped the simulation operate realistically and successfully, and supported the team's understanding of the system.

### Vanessa Kurniawidi

Vanessa created the simulation environment and contributed to the project page, including the system logic flowchart and theoretical analysis. She also helped debug the simulation to improve robustness and task execution.

## Install and run

Developed on Windows with Python 3.12. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ".\Maryam_Final_Integration\requirements.txt"
.\.venv\Scripts\python.exe ".\Maryam_Final_Integration\main_final_integration.py"
```

Enter an order such as `r,b,y` or `blue,yellow,red` at the prompt. Each colour can appear at most once. Alternatively, supply the order directly:

```powershell
.\.venv\Scripts\python.exe ".\Maryam_Final_Integration\main_final_integration.py" b,y,r --seed 7
# Static comparison: disable platform motion.
.\.venv\Scripts\python.exe ".\Maryam_Final_Integration\main_final_integration.py" r,b,y --red-amplitude 0
# Optional numerical diagnostics.
.\.venv\Scripts\python.exe ".\Maryam_Final_Integration\main_final_integration.py" r,b,y --debug
```

Dependencies are listed in the integration folder's `requirements.txt`: NumPy, MuJoCo, OpenCV, SciPy and Matplotlib. Keep its `universal_robots_ur3e/` scene and mesh assets alongside the application.

## What the demonstration shows

1. The MuJoCo viewer and wrist-camera window open, and the robot settles at HOME.
2. Before each pickup, the arm moves the wrist camera to find and measure the matching tray, then returns to view the cubes.
3. The robot detects the requested cube, approaches, closes the gripper and lifts it. Fresh camera observations and finger position/force readings must confirm pickup.
4. It transfers the cube to the measured tray position, releases it and raises the camera to inspect placement. Confirmation requires four fresh observations within the tray interior and settled height range, with the gripper open.
5. After confirmed placement, it returns through a clear waypoint to HOME before processing the next cube. The terminal reports progress and the final number of confirmed deposits. An unverified task outcome stops the sequence.

Press `q` in the camera window to stop. The red platform moves side to side by +/-20 mm with a 20-second cycle by default. It continues through closure and lift; the robot does not deliberately wait for a programmed pause or stop the cube to grasp it.



## Sensor feedback and evaluation

Noisy simulated joint encoders feed a joint Kalman observer. Estimated joints provide tool/camera poses through forward kinematics. RGB-D detections and object Kalman filters provide cube positions; measured tray positions provide placement references. Known tray regions direct the camera search, rather than supplying the final drop coordinates. Pickup and retention decisions use camera evidence and simulated finger position/force sensors.

Simulator state is used to construct the scene, generate simulated sensor readings and drive the moving platform. Separate ground-truth positions, contacts and velocities are logged for evaluation and do not supply the task's object targets or arm state feedback. The controller uses local linear models; local controllability and stability checks do not guarantee global stability under contact or actuator limits.

## Logs and checks

New runs write to `Maryam_Final_Integration/results/latest/` by default. Use a distinct `--output results/run_name` for each experiment to preserve earlier logs. Outputs include `summary.json`, `trajectory.csv`, `vision.csv`, `object_tracking.csv`, `gripper.csv` and `last_camera.png`. Simulation time and wall-clock execution time are different measures. Full generated logs are ignored by Git; selected evidence is committed under the integration documentation.

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ".\Maryam_Final_Integration\tests" -v
.\.venv\Scripts\python.exe ".\Maryam_Final_Integration\main_final_integration.py" --headless --smoke-seconds 1 --output results/smoke
```

Headless rendering still requires an OpenGL-capable environment. A smoke check only holds HOME; it does not demonstrate successful sorting. See the recorded validation report for camera validation, controller comparisons and experiment evidence.

## Acknowledgements

<!-- Team: add references, asset attribution and acknowledgements here. -->

## Generative AI declaration

ChatGPT and Claude were used to assist with technical explanations, code development, debugging and documentation. Team members are responsible for reviewing AI-assisted material, checking it against the implementation and results, and understanding the submitted system.

<!-- Team: complete the declaration with all other tools used, their purposes and any required subject-specific details. -->
