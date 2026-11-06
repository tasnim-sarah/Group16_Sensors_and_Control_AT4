# Explaining the sorting robot

## A short explanation you can give aloud

"Our UR3e uses a wrist-mounted simulated RGB-D camera to find a coloured cube and its matching tray. Depth converts the image detection into a 3D measurement. Kalman filters estimate object motion and robot joint states. Position-based visual servoing and inverse kinematics generate joint targets, and LQR tracks those targets using estimated joint positions and velocities. After a lift, camera observations and gripper sensors must confirm pickup. We monitor for slipping and visually verify placement. Simulator truth is logged separately to evaluate whether the sensor decisions were correct."

Read the implementation and explain it in your own words. This describes the integrated system, not a claim that you personally authored every component.

## What happens on screen, in order

1. **Scene setup.** Load the UR3e, table, cubes, trays, gripper and fixed wrist camera. A random seed generates reproducible cube positions. By default, the red cube rides a slowly moving physical platform with a 4 cm side-to-side sweep; blue and yellow are stationary. Enter the desired order at the terminal prompt, or pass it on the command line.
2. **Model and observer setup.** Build a local actuator model, check controllability/observability, solve LQR and initialise the joint observer. Small model-validation perturbations run on copied simulator data; they are not extra sorting trials.
3. **HOME settling.** Hold the starting pose for at least 0.5 simulated seconds. Establish feedback and initial camera measurements.
4. **Pre-pick destination search.** Move through a clear waypoint, then view the requested tray. The expected tray region directs the camera search; the drop reference is a measured RGB-D position. Return to a viewpoint for the cubes. This explains the apparent "practice run": the robot is looking before grasping. The gripper stays open; no cube is counted as picked or placed.
5. **Cube acquisition and approach.** Detect the requested colour, distinguish cube geometry from the larger same-colour tray, and estimate its world position. Move above it at approximately 0.14 m TCP height.
6. **Descent and closure.** Reacquire the cube position, descend to approximately 0.03 m TCP height, close the fingers and allow contact to settle. Static cubes use a fixed acquired XY reference. Red uses live Kalman position/velocity prediction during descent and closure, with at most 3 s prediction through final occlusion. Its platform is not stopped at closure.
7. **Verification lift.** Lift to approximately 0.16 m. Require a measured cube rise of at least 35 mm, a stable cube-to-tool offset over four fresh camera frames, and supporting finger displacement/force. Occlusion is unconfirmed, not a successful grasp.
8. **Transfer.** Move over the measured tray at approximately 0.22 m. Check gripper retention and visible relative-offset drift; abort if the cube appears to slip.
9. **Release.** Descend to approximately 0.11 m, open the fingers and let the cube settle.
10. **Placement inspection.** Raise the tool for a clearer view. Require four fresh observations of the cube within a conservative tray interior and settled height range, with the fingers open.
11. **Return and repeat.** Return through the clear waypoint and HOME. Repeat for the next requested cube. There is no hidden successful trial before the requested sequence.
12. **Evaluation.** Save measured performance, sensor confirmations, independent truth scoring, uncertainties, control commands, torques and failure causes.

Heights refer to the tool centre point (TCP), not the wrist or cube centre. Engineering thresholds are initial design choices with limited validation, not universal guarantees.

## What runs continuously

| Component | Purpose | Current implementation |
|---|---|---|
| Physics and joint feedback | Simulate motion and correct tracking error | 0.002 s steps, approximately 500 Hz |
| Noisy joint encoder | Simulated sensor input | Default standard deviation 0.0005 rad |
| Joint Kalman observer | Estimate six angles and six velocities | Uses previous applied actuator command and measured angles |
| Forward kinematics | Calculate tool/camera poses | Uses estimated joints in separate kinematic data |
| RGB-D detection | Measure object locations | Approximately 15 Hz in simulation time |
| Object Kalman filters | Smooth position and estimate velocity | Timestamp-dependent constant-velocity model |
| PBVS | Generate short Cartesian approach targets | Based on measured 3D position error |
| Inverse kinematics | Convert Cartesian pose to joint targets | Damped numerical solution; does not command motors |
| LQR | Track joint targets with state feedback | All six arm actuators use estimated-state feedback |
| Model refresh | Keep local dynamics relevant as posture changes | Approximately every 0.5 simulated seconds |

Processing can run slower than real time. Do not confuse camera wall-clock FPS with simulated capture intervals.

## How to defend the main design decisions

**Why LQR?** It provides a systematic state-feedback design balancing tracking error, joint velocity and actuator-target variation through Q and R. We tested alternative settings rather than only stating weights. It is a local design and does not guarantee global stability with contact or saturation.

**Why do we also have PBVS and IK?** LQR needs joint targets. PBVS decides the next small Cartesian target from vision; IK converts that target into joint angles. Neither replaces the arm's LQR actuator feedback. MuJoCo's internal position servos are the simulated actuator plant; we do not claim torque LQR or that LQR controls the gripper.

**Why a 12-state model?** The six arm joints each have an angle and velocity. Encoders measure angles, while the observer estimates velocities from how angles evolve and the actuator model. Initial controllability and observability ranks are both 12/12 for the local model.

**Why two types of Kalman filter?** Joint estimation and object tracking are different problems. The joint filter estimates arm state from encoders and actuator dynamics. Each object filter estimates 3D position and velocity from camera measurements.

**Where do Q and R come from?** For the joint KF, R follows the deliberately injected encoder variance; Q accounts for model mismatch. For LQR, Q weights state errors and R weights actuator-target deviations. KF Q/R and LQR Q/R have different meanings. Initial process values and vision thresholds are engineering choices; the local experiments show trade-offs but do not establish universal optimal tuning.

**How does an image become a robot target?** Colour segmentation finds pixels. Depth and camera intrinsics back-project them into camera coordinates. Estimated-joint FK and the fixed camera mounting transform those coordinates into the world frame. Known cube size supplies an approximate surface-to-centre correction.

**How can a red cube and red tray be distinguished?** Same-colour objects require geometry as well as colour. The camera uses contour nesting and measured horizontal extents to separate a compact cube from an approximately 110 mm tray rim. Neutral tray floors keep the cube visually separate from the coloured walls after placement. The detector does not read simulator object IDs to classify them.

**Why inspect the tray first?** We avoid picking a cube before knowing where we can place it. The wrist camera cannot see every region at once. Known workspace regions guide search, but the measured tray position supplies the drop target. Searching before every pick is conservative and adds time; caching reliable static-tray measurements is a future optimisation.

**How do we know pickup happened?** A closing gripper alone is insufficient. We require several fresh visual measurements showing lift and consistent relative position, plus gripper evidence. This can produce conservative false negatives during occlusion; it must not silently declare success.

**Where is simulator truth allowed?** Scene setup, sensor generation, optional environment motion and independent evaluation. Robot feedback uses estimated joints, and task confirmation uses sensors. Reading simulated encoders is legitimate sensor simulation; using exact cube coordinates as the grasp target would not be.

**Is the grasp attached or welded?** No. Frictional contacts hold the cube. Finger geometry and contact solver settings were revised after observed slipping. Their individual effects were not isolated, so we must not attribute the improvement to one setting alone.

**What evidence can we claim?** Read each JSON's configuration and source hashes. Earlier repeated trials used the older tray palette. Colour/geometry changes require new validation; never present those earlier successes as tests of the new detector. Local controller experiments are not full sorting robustness trials.

## What changed in this revision of the project

- The former observer/LQR calculations were connected to actual arm feedback through a model compatible with position actuators.
- Robot and camera kinematics now use estimated joints.
- The camera mounting stays fixed during a trial; the arm moves to search viewpoints.
- Pickup, carry and placement require sensor evidence, with explicit failure outcomes.
- Grasp contact geometry and solver settings were revised after measured slipping.
- Seeded runs, raw logging, calibration benches, tuning comparisons and report figures were added.
- Tray walls now match their cube colours; geometry distinguishes them.

Team contributions must describe each person's real work and review. The AI declaration must acknowledge the assistance used. For the viva, be ready to point to the relevant functions and explain a failed run as well as a successful one.

## Demonstrating moving-target tracking

Start with `python main_final_integration.py`, then enter `r,b,y`. The red cube is already moving during the tray search and approach. Watch the colour-tagged terminal milestones and the active-target camera overlay. Use `--debug` to see raw/filtered XY, estimated velocity, observation age and the evaluation-only pre-contact speed. Detailed tracking messages are scoped to the active cube and stop after placement. The target used by the controller comes from the camera filter, not that evaluation value or the platform trajectory.

The moving platform is an environment actuator, separate from the six arm actuators controlled by LQR. It runs continuously; the cube stops following it only when physical fingers lift it away. Do not describe this as a cube pose being animated until a scripted stop. Known failure modes include partial-view bias and occlusion. Existing historical results must not be relabelled as tests of this new configuration.

## Useful places to look in the code

- `robot_scene.py`: scene, tray palette, fixed camera mounting and gripper.
- `main_final_integration.py`: `locate_tray`, `pick_place`, `step`, `capture`, `ik`, `schedule`.
- `controlfile.py`: local state-space model, observer and LQR.
- `cameramodulekal.py`: colour/geometry detection, depth conversion and object filtering.
- `visual_verification.py`: pickup and placement evidence rules.
- `docs/index.html`: recorded evidence and limitations.

The earlier circular moving-stage revision completed both a red-first moving pick and the complete blue/yellow/red sequence for seed 7. The red pre-contact horizontal speed was approximately 3.142 mm/s in both runs. Evidence is in `results/live_red_seed7_side_view/` and `results/live_order_byr_side_view/`; the validation report distributes selected logs and figures.

The current default is side-to-side Y motion, +/-20 mm with a 20 s period. Natural direction reversals are not scripted grasp pauses. The table is matte grey, and normal terminal output hides filter ages and coordinates. The previous yellow hesitation was a destination search: the saved log showed detections every camera frame during approach/descent and confirmed placement. Yellow colour thresholds remain unchanged.

Default terminal example: `[YELLOW] Searching for cube`, `[YELLOW] Cube detected - live tracking active`, `[YELLOW] Pickup confirmed`, `[YELLOW] Placement confirmed - tracking finished`. Detailed per-second output is disabled unless `--debug` is requested and is scoped to the active cube. Physical motion continues independently of whether its diagnostics are displayed.

The grey-table side-to-side revision passed full `r,b,y` and `b,y,r` sequences on seed 7. Each logged 40.0 mm of observed red travel before grasp; red pre-contact speeds were 2.675 and 5.547 mm/s. Eleven current behaviour tests pass. This supersedes the default motion and logging descriptions of the earlier circular prototype.
