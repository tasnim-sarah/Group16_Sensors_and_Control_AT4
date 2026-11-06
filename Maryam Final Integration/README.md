# Maryam Final Integration

**Run only `main_final_integration.py`.** The other four Python files beside it support the controller, camera, scene and pickup/placement verification.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe ".\Maryam Final Integration\main_final_integration.py"
```

Enter an order such as `r,b,y` and press Enter. Red moves side to side, the table is grey, and the terminal reports the active cube. Add `--debug` only for numerical diagnostics.

Folder contents:

- `main_final_integration.py`: the application to run.
- `controlfile.py`, `cameramodulekal.py`, `robot_scene.py`, `visual_verification.py`: required support code.
- `universal_robots_ur3e/`: the complete simulation scene and mesh assets.
- `docs/`: walkthrough, measured report and selected evidence.
- `tools/` and `tests/`: optional evaluation tools and behaviour tests.
- `results/`: preserved full run logs, ignored by Git.

For the detailed commands below, first change into this folder:

```powershell
cd "Maryam Final Integration"
```

# Group 16: UR3e vision-guided sorting

41014 Sensors and Control, Assessment 4, Track B. Simulation only.

Team: Tasnim Sarah, Maryam Qutub, Princess Kisakye, Vanessa Kurniawidi.

## Matching-colour trays and walkthrough

The canonical scene now uses red, yellow and blue tray walls to match the cubes. Neutral grey floors preserve visual separation after placement. Colour plus RGB-D geometry distinguishes a compact cube from a larger tray rim; colour alone would confuse them. Historical result files use their original palette and source hashes.

See [the step-by-step explanation and viva preparation guide](docs/pipeline_walkthrough.md) for what happens on screen and how to explain the design. The initial trip to the tray is a **pre-pick camera search**, not a practice sorting trial. Console stage labels now make this explicit.

The earlier matching-colour revision was covered by five automated tests, including rendered RGB-D checks for empty trays and cubes inside each matching-colour tray. Its new complete seed-7 run is recorded in `results/matching_trays_seed7_partial_fix/`; older palette trials remain separate evidence.

The earlier geometry detector accepted 40/45 observations. The current grey-table side-view bench accepts 45/45 static grid observations, with about 2.42 mm 3D RMSE and 3.58 mm maximum error. These are static sensor-bench results, not a moving-grasp success rate. Historical tests retain their own source hashes and must not be relabelled as current-revision evidence.

## Current status

This folder contains Maryam's maintained integration. The original project files outside this folder are restored separately; older revisions are available in Git history.

Estimated-state LQR actuator feedback and sensor-based verification are implemented. Behaviour checks pass for a small joint step, missed/occluded pickups and placement decisions. A static yellow-cube run (seed 7) completed pickup, transfer, visual placement confirmation and return; independent simulator scoring agreed. The corrected pipeline completed all three cubes for seeds 7, 11 and 23; seed 23 ran in a fresh virtual environment. This small selected sample is encouraging, but broader sorting reliability remains under evaluation.

## Terminal order and continuous live target

Run `python main_final_integration.py` without colours to enter the order in the terminal, for example `r,b,y`, `y r b` or `blue,yellow,red`. Invalid/duplicate colours are rejected and the prompt repeats. Each requested cube must be picked, deposited and returned from before the next starts. Explicit command-line colours skip the prompt; headless runs never wait for input.

The red cube moves by default. A physical platform beneath it travels side to side along Y, +/- 20 mm (4 cm total), with a 20 s cycle. Peak commanded speed is about 6.28 mm/s. It slows smoothly and reverses at the ends, without a programmed pause before grasping. Friction carries the free cube; its pose and velocity are not overwritten. The platform continues moving during closure and lift. Once lifted, the cube follows gripper contact physics and comes to rest after placement. Motion is not artificially stopped when the gripper closes.

The red approach, descent and closure use Kalman position/velocity prediction. Partial views are rejected before grasp, and final descent/closure permit at most 3 s of predicted tracking through occlusion. Fresh visual evidence is still required to confirm pickup. The fixed camera views between the fingers to improve retained-cube visibility. The kinematic/linearisation workspace excludes external cubes, avoiding fictitious contacts from their default XML positions; the real scene retains contacts normally.

The earlier circular-motion revision passed a red-first moving pick and a complete `b,y,r` sequence on seed 7. All pickups and placements were visually confirmed and agreed with independent scoring. In both tests, measured red speed immediately before first finger contact was approximately 3.142 mm/s. That revision passed nine automated tests, including order parsing, prompt retry and physical platform transport. These selected trials do not establish all-seed reliability.

By default, the terminal shows only milestones for the active cube: matching tray found, cube detected, pickup confirmed and placement confirmed. Camera overlays follow the active cube rather than always showing red. Add `--debug` for coordinates, estimated velocities, model details, observation ages and the full results JSON. Red diagnostics stop after placement; blue and yellow report their own detections. `object_tracking.csv` records estimates, uncertainty and evaluation truth. A separately labelled evaluation event records actual cube speed just before first finger contact; this never controls the robot.

```powershell
..\.venv\Scripts\python.exe main_final_integration.py
# At the prompt: r,b,y
..\.venv\Scripts\python.exe main_final_integration.py b,y,r --seed 7
# Optional numerical diagnostics:
..\.venv\Scripts\python.exe main_final_integration.py r,b,y --debug
# Explicit static comparison:
..\.venv\Scripts\python.exe main_final_integration.py r,b,y --red-amplitude 0
```

The current grey-table, 4 cm sweep revision completed both `r,b,y` and `b,y,r` on seed 7, with all six pickups and placements confirmed and independently scored correct. Recorded red travel before grasp was 40.0 mm in both runs; pre-contact speeds were 2.675 and 5.547 mm/s respectively. Evidence is in `results/clear_sweep_rby_seed7/` and `results/clear_sweep_byr_seed7/`. These two order tests do not establish all-seed reliability.

Eleven current behaviour tests pass, including the 4 cm side-to-side path and target-specific diagnostic suppression. Detailed CSV/JSON logging is preserved even when normal terminal output is quiet.

Use `--red-amplitude` for half of the side-to-side travel in metres and `--red-period` for the complete cycle in seconds. The table is matte neutral grey to improve colour contrast. The latest pre-change user log already showed yellow detected throughout approach/descent and confirmed placement; the extra motion was the tray search, so yellow thresholds were not weakened. Historical experiments retain their original motion settings and source hashes. Batch studies explicitly disable platform motion except for the moving-target condition.

## Install and run (Windows PowerShell)

```powershell
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe main_final_integration.py yellow --seed 7
```

Start with one static cube. Three-cube sequence:

```powershell
..\.venv\Scripts\python.exe main_final_integration.py red blue yellow --seed 7 --output results/sequence7
```

Use `--headless` for automated runs (offscreen OpenGL is still required). Commands must run from the repository root. Use `--help` for tuning options. On macOS the interactive MuJoCo viewer requires `mjpython`.

Control smoke check and behaviour tests:

```powershell
..\.venv\Scripts\python.exe main_final_integration.py --headless --smoke-seconds 1 --output results/smoke
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

A smoke check holds HOME; it does not count as a sorting success. Failed task stages return a nonzero exit code.

## Feedback and model

The sole implemented arm actuator feedback controller is discrete LQR. RGB-D position-based visual servoing (PBVS) generates short Cartesian targets; damped numerical inverse kinematics converts these into joint references. IK does not issue actuator commands. MuJoCo's existing internal position servos remain the simulated actuator plant, so this is not torque LQR.

State: x = [q1..q6, dq1..dq6] (12 states). Input: u = six actuator position targets in radians. Output: y = six noisy encoder angles. A is 12x12, B is 12x6, C = [I6, 0] is 6x12, D is 6x6.

The local affine model is x[k+1] = A x[k] + B u[k] + c. Finite differences of the actual MuJoCo transition yield A and B. c makes the prediction match the nominal step. The model is relinearised every 0.5 simulated seconds around estimated joints and model-predicted gravity equilibrium, without reading true arm joints. Coupled dynamics, contact, force limits and large posture changes limit local validity. A small perturbation validation reports a local residual; that alone does not validate global dynamics.

At initialisation, measured ranks are controllability 12/12 and observability 12/12. This applies to the local actuator model. The encoder output observes velocity through time evolution. Each run saves ranks and its local validation residual; `--debug` also prints the model report.

LQR minimises sum(e^T Qc e + du^T Rc du), with default Qc = diag(20 I6, 1 I6), Rc = 0.1 I6. The command is u = u_equilibrium(q_ref) - K [q_hat-q_ref, dq_hat], clipped to actuator bounds. Each local design checks that the spectral radius of A-BK is below one. This local check does not guarantee stability under saturation or contact. Weight comparisons are still needed to justify the defaults empirically.

The joint Kalman filter predicts with the applied previous actuator target and affine model, and corrects using noisy encoders. Default encoder standard deviation is 0.0005 rad, R = sigma^2 I6 and Q = 1e-5 I12. Joseph covariance updates preserve numerical covariance quality. The uniform process covariance is an initial model-mismatch choice, not a measured optimum. Separate process scaling for angles and velocities is a possible improvement.

## Camera and frames

The camera is rigidly wrist-mounted. Its virtual mounting and 75 degree vertical field of view are defined once during scene construction. It is never re-aimed by changing calibration during a trial. The arm moves to search viewpoints.

Pixel + depth -> camera coordinates -> estimated wrist-camera pose -> world coordinates -> Cartesian target. For MuJoCo's camera frame, p_camera = [(u-cx)d/fx, -(v-cy)d/fy, -d]. With square pixels, fx = fy = height/(2 tan(fovy/2)). The measured point is transformed as p_world = R_world_camera p_camera + t_world_camera. Camera extrinsics come from forward kinematics of estimated joints, using a separate MjData object.

Colour masks and median depth localise each object. The known 40 mm cube geometry supplies an approximate 20 mm surface-to-centre correction along the viewing ray. This approximation can create angle-dependent bias and requires multi-location validation. Simulator intrinsics are known; this is not evidence of physical camera calibration.

Object Kalman filters have [position, velocity] states, timestamp-dependent transition intervals, R = 2.5e-5 I3 and acceleration-noise standard deviation 0.07 m/s^2. Missed measurements are not fresh visual evidence.

## Pickup and placement decisions

After closure, a short lift must show the cube rising at least 35 mm and remaining near the TCP, with a stable cube-to-tool offset over four consecutive fresh frames. Finger opening and force provide supporting sensor evidence. Missing vision gives **unconfirmed**, not successful pickup. These thresholds are initial engineering settings and need measured false-positive/false-negative evaluation.

Placement requires an open gripper and four fresh observations of the cube centre within a conservative tray interior and settled height range. Sensor confirmation and independent simulator placement scoring are logged separately. The system stops when it cannot verify the required outcome.

Ground truth is used for trial setup, simulated encoder/finger readings, optional imposed target motion and evaluation metrics only. `truth_*` log columns never feed control or task decisions. The camera renders the actual scene, as a simulated sensor should.

## Reproducible experiments

```powershell
..\.venv\Scripts\python.exe tools/run_batch.py --seeds 7 11 23 --conditions nominal encoder_x4 depth_5mm --output results/study01
```

Additional matched-seed conditions compare LQR effort/tracking weights, observer process noise and moving targets. Batch runs preserve failures/timeouts in the denominator and refuse to overwrite an existing run folder. Expand to 10-20 seeds per primary condition once nominal behaviour is stable.

`--red-amplitude 0.03` selects 6 cm total side-to-side travel. The platform continues through grasp; the cube separates through normal contact physics when lifted. This is a bounded moving-stage experiment, not an industrial conveyor validation.

Each run produces `summary.json`, `trajectory.csv`, `vision.csv`, and `last_camera.png`. Trajectory data include true and estimated joints, references, uncertainty, commands, torques and saturation. Joint RMSE is radians; torque-squared integral is N^2 m^2 s and is an effort proxy, not energy. Separate capture/detection coverage must accompany localisation RMSE.

## Assessment evidence still required

- Repeated full-sequence successes under varied seeds/noise, with failures diagnosed.
- Quantitative camera localisation validation and estimator/controller tuning comparisons.
- Plots of estimates with uncertainty against evaluation truth, tracking and control effort.
- Completed project page in `docs/index.html`, accurate per-member contributions and a five-minute video link. The old project page still has placeholders and must not be treated as verified evidence.

No grade is guaranteed by implementation alone.

## Measured development evidence

Open [the validation report](docs/index.html) for the recorded trials, raw-data links and plots. The three corrected nominal full-sequence trials achieved 3/3 successes, with visual confirmation and independent placement scoring agreeing for all nine cubes. The seeds are selected development trials, not an unbiased reliability estimate. A fresh dependency installation passed all four behaviour tests and the complete seed-23 sequence. A separate seed-7 run with 0.002 rad encoder noise and 0.005 m per-pixel depth noise also completed all three cubes, with all visual decisions agreeing with evaluation scoring. This is one combined-noise run, not repeated evidence for each noise source independently.

The successful yellow-cube trial is recorded in `results/yellow_contact_revision/`: 78.204 simulated seconds, joint tracking RMSE 0.02432 rad, joint estimation RMSE 0.000488 rad and no actuator-target clipping. Its source hashes identify the tested revision. This is one successful trial, not a reliability estimate.

The final static camera bench measured all 45 observations across nine locations. With nominal encoder noise, 3D RMSE was 6.39 mm and the largest error was 19.08 mm; with 5 mm added per-pixel depth noise, RMSE was 6.55 mm. Median depth aggregation reduces independent pixel noise, so the small change does not imply robustness to correlated depth bias. These results use the final camera/contact geometry; further revisions require revalidation.

Thirty local joint-step experiments compare five settings, two encoder noise levels and three matched seeds. At 0.0005 rad encoder noise, increasing LQR position weight from 20 to 80 reduced mean tracking RMSE from 0.004227 to 0.003047 rad (about 28%), while torque-squared integral rose from 284.29 to 310.97 N² m² s (about 9%). With encoder noise 0.002 rad, the same weight increase raised effort from 313.88 to 503.94 (about 61%) for a similar tracking improvement. The default weight 20 is therefore retained pending integrated task comparisons.

Reducing observer process covariance from 1e-5 I to 1e-7 I reduced local angle estimation RMSE from 0.000484 to 0.000271 rad, but increased velocity estimation RMSE from 0.00193 to 0.00700 rad/s. This is evidence of a smoothing/responsiveness trade-off, not proof that either setting is universally optimal. The local bench stays near HOME and cannot substitute for full-task trials.

```powershell
..\.venv\Scripts\python.exe tools/validate_camera.py --output results/camera_study
..\.venv\Scripts\python.exe tools/evaluate_control.py --output results/control_study
..\.venv\Scripts\python.exe tools/plot_results.py results/control_study
..\.venv\Scripts\python.exe tools/plot_results.py results/sequence7
```

Early transfer failures were retained: increasing finger position gain alone did not prevent slipping. Longer pads and stiffer contact settings with the elliptic friction solver produced the successful trial, using physical contact without welds. These changes were combined, so their individual effects have not been isolated. Continuous visual offset and gripper-sensor checks now stop a transfer when slip is detected.

## Generative AI declaration

OpenAI Codex assisted with code review, implementation, testing infrastructure and documentation during the LQR/visual-verification revision. Team members must review the generated changes, record their actual contributions and be able to explain the whole pipeline in the viva. Add any other tools used by the team and their purposes.

Raw generated logs remain in the ignored `results/` folder (they are large). Selected JSON summaries, study CSVs, figures and a downsampled seed-7 trajectory are distributed in `docs/static/evidence/`. Run `python tools/build_validation_report.py` after new experiments to refresh the report.

Historical result JSON files and source hashes are preserved exactly as recorded; their original filenames refer to the earlier repository layout. New runs record `main_final_integration.py`. The generated verification environment and caches were relocated to ignored `results/_local_generated/` because automatic deletion was blocked.
