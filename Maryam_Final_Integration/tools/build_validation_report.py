"""Publish measured local evidence into docs; never invent missing result rows."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import html
import gzip
import csv
import json
import shutil
from pathlib import Path

ROOT=PROJECT_ROOT
DEST=ROOT/'docs'/'static'/'evidence'


def archive(directory):
    src=ROOT/'results'/directory
    if not src.exists():return
    target=DEST/directory
    target.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','force_limits.json','tracking.png','uncertainty.png','localisation.png','tradeoffs.png','live_tracking.png','object_tracking.csv','trials.csv','measurements.csv','last_camera.png'):
        if (src/name).exists():shutil.copy2(src/name,target/name)
    if directory=='sequence_seed7_detection_fix' and (src/'trajectory.csv').exists():
        with (src/'trajectory.csv').open(newline='') as source, gzip.open(target/'trajectory_50hz.csv.gz','wt',newline='') as destination:
            reader=csv.reader(source);writer=csv.writer(destination);writer.writerow(next(reader))
            for index,row in enumerate(reader):
                if index%10==0:writer.writerow(row)


def summary(directory):
    path=ROOT/'results'/directory/'summary.json'
    return json.loads(path.read_text()) if path.exists() else None


def main():
    runs=[('Current grey table, side-to-side motion, red/blue/yellow order','clear_sweep_rby_seed7'),
          ('Current grey table, side-to-side motion, blue/yellow/red order','clear_sweep_byr_seed7'),
          ('Earlier circular target, red first','live_red_seed7_side_view'),
          ('Earlier circular target, blue/yellow/red order','live_order_byr_side_view'),
          ('Matching-colour trays, earlier static geometry detector','matching_trays_seed7_partial_fix'),
          ('Single yellow, development milestone','yellow_contact_revision'),
          ('Nominal, corrected detection','sequence_seed7_detection_fix'),
          ('Nominal, corrected detection','sequence_seed11_detection_fix'),
          ('Nominal, fresh installation','fresh_sequence_seed23'),
          ('Encoder 2 mrad + pixel depth 5 mm','sequence_seed7_noisy')]
    development=[('Moving-target prototype, stale final camera reference','live_red_seed7'),
                 ('Moving target, earlier camera hid retained cube','live_red_seed7_contact_free_model'),
                 ('Matching-colour prototype, partial face rejected','matching_trays_seed7'),
                 ('Earlier detector','sequence_seed7'),('Earlier detector','sequence_seed11'),
                 ('Earlier detector','sequence_seed23'),('Earlier gripper','yellow_grip2000')]
    camera_dirs=['final_calibration_clean','final_calibration_5mm','matching_camera_partial_fix','live_side_view_camera_validation','clear_sweep_camera_validation']
    for _,directory in runs+development:archive(directory)
    for directory in camera_dirs+['local_control_study']:archive(directory)
    def run_table(records):
        rows=[]
        for label,directory in records:
            s=summary(directory)
            if s is None:
                rows.append(f'<tr><td>{html.escape(label)}</td><td colspan="6">Not yet recorded</td></tr>');continue
            seed=s['config']['seed'];requested=len(s['config']['order'])
            failure=html.escape(s.get('failure') or '—')
            rows.append(f'<tr><td>{html.escape(label)}</td><td>{seed}</td><td>{s["completed"]}/{requested}</td>'
                f'<td>{"PASS" if s["sequence_success"] else "FAIL"}</td><td>{s["pickup_confirmations"]} / {s["placement_confirmations"]}</td>'
                f'<td>{failure}</td><td><a href="static/evidence/{directory}/summary.json">JSON</a></td></tr>')
        return '<div class="scroll"><table><thead><tr><th>Condition</th><th>Seed</th><th>Completed</th><th>Sequence</th><th>Visual pickup / place</th><th>Failure</th><th>Evidence</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    camera_rows=[]
    for directory in camera_dirs:
        s=summary(directory)
        if s:
            camera_rows.append(f'<tr><td>{html.escape(directory)}</td><td>{s["depth_noise_std_m"]*1000:.0f} mm</td><td>{s["detected"]}/{s["attempted"]}</td>'
                f'<td>{s["rmse_3d_m"]*1000:.2f} mm</td><td>{s["max_error_m"]*1000:.2f} mm</td>'
                f'<td><a href="static/evidence/{directory}/measurements.csv">Measurements</a></td></tr>')
    study=summary('local_control_study')
    control_rows=[]
    if study:
        for s in study['results']:
            control_rows.append(f'<tr><td>{s["condition"]}</td><td>{s["encoder_noise_std_rad"]*1000:g}</td>'
                f'<td>{s["tracking_rmse_rad"]*1000:.3f}</td><td>{s["estimation_rmse_rad"]*1000:.3f}</td>'
                f'<td>{s["velocity_estimation_rmse_rad_s"]*1000:.3f}</td><td>{s["torque_squared_integral"]:.1f}</td></tr>')
    document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Group 16 — measured validation</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f5f6fa;color:#182337;margin:0}main{max-width:1060px;margin:auto;padding:32px 20px}section{background:white;padding:24px;margin:20px 0;border:1px solid #dde2ec;border-radius:10px}h1,h2{line-height:1.25}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}th{background:#eef2f8}.scroll{overflow:auto}img{max-width:100%;height:auto}pre{overflow:auto;background:#eef2f8;padding:16px}a{color:#174ba4}.notice{border-left:5px solid #d29416;padding-left:16px}small{color:#556}
</style></head><body><main><h1>Group 16: measured validation</h1>
<p>UR3e vision-guided sorting · Track B · Simulation evidence · 6 November 2026</p>
<p class="notice">This is a development evidence report, not a grade claim. The current scene uses matching red/yellow/blue tray walls and neutral floors; older rows retain their original palette and source hashes. <a href="pipeline_walkthrough.md">Read the stage-by-stage walkthrough and viva guide.</a> Video, accurate individual contributions and the final project-page narrative still require completion. <a href="../../docs/index.html">Return to the submission draft.</a></p>
<section><h2>Implemented pipeline</h2><pre>RGB-D → object Kalman filters → PBVS Cartesian targets → inverse kinematics
      → joint references → LQR → MuJoCo position-servo actuators
Noisy encoders → joint Kalman observer → estimated joints → LQR and FK
Fresh RGB-D + finger sensors → pickup, carry and placement decisions
Simulator truth → evaluation logs only</pre>
<svg viewBox="0 0 900 190" role="img" aria-label="Coordinate frame diagram" style="width:100%;height:auto"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#174ba4"/></marker></defs><g fill="#eef2f8" stroke="#174ba4"><rect x="10" y="55" width="130" height="55" rx="6"/><rect x="210" y="55" width="170" height="55" rx="6"/><rect x="460" y="55" width="140" height="55" rx="6"/><rect x="690" y="55" width="170" height="55" rx="6"/><path d="M140 82H210M380 82H460M600 82H690" marker-end="url(#arrow)"/><path d="M295 110V155H460" marker-end="url(#arrow)"/></g><g text-anchor="middle" font-family="system-ui" font-size="16" fill="#182337"><text x="75" y="88">World / base</text><text x="295" y="88">Wrist frame</text><text x="530" y="88">Camera frame</text><text x="775" y="88">RGB-D pixels</text><text x="175" y="43">FK(q_est)</text><text x="420" y="43">Fixed mount</text><text x="645" y="43">K, depth</text><text x="565" y="160">TCP frame (fixed tool transform)</text></g></svg><p>Forward kinematics supplies world-to-wrist pose from estimated joints. Fixed wrist-to-camera and wrist-to-TCP transforms then define sensor and tool poses. Pixels are back-projected with K and depth, then transformed in the reverse direction into world coordinates. MuJoCo camera axes are +x right, +y up and -z optical forward. <a href="static/evidence/camera_setup.json">Numerical camera mounting and intrinsics</a>.</p><p>LQR is the only implemented arm actuator feedback controller. IK generates references and the simulated actuator plant retains its internal position servos. A fixed wrist-camera mounting is used throughout each trial. Known tray locations direct a search; RGB-D measurements supply the drop target.</p></section>
<section><h2>Recorded task trials</h2><p>Each row is a complete requested run, including failure. These development seeds are a small, selected sample, so they cannot establish a general reliability rate. Source hashes in each JSON distinguish revisions.</p>'''+run_table(runs)+'''
<p><a href="static/evidence/sequence_seed7_detection_fix/trajectory_50hz.csv.gz">Download the corrected seed-7 joint trajectory (CSV gzip, downsampled to 50 Hz)</a>. Full 500 Hz logs remain in the local results folder; downsampling is only for distribution.</p><h3>Earlier failures, retained for diagnosis</h3>'''+run_table(development)+'''
<p>Earlier blue-cube failures came from colour thresholds and background selection. Large blue floor contours could suppress a valid object before workspace rejection. Workspace validation now precedes candidate selection, and a regression test recreates that failure. Earlier grasp slip persisted after increasing finger gain alone; longer pads, stiffer contact and the elliptic friction solver produced a successful transfer. Their individual effects have not yet been isolated.</p></section>
<section><h2>Observed tracking and uncertainty</h2><p>The successful yellow-cube development trial records references, evaluation truth, estimates and covariance. The separate error plot makes the small uncertainty band visible. These are measurements from the recorded run, not illustrative curves.</p><img src="static/evidence/yellow_contact_revision/tracking.png" alt="Recorded joint tracking and torque"><img src="static/evidence/yellow_contact_revision/uncertainty.png" alt="Recorded joint estimation errors and two-sigma uncertainty"></section>
<section><h2>Model and feedback</h2><pre>x = [q₁ … q₆, q̇₁ … q̇₆]ᵀ       y = [I₆  0] x + v
x[k+1] = A x[k] + B u[k] + c
A: 12×12     B: 12×6     C: 6×12     D: 6×6
u = actuator joint-position targets (rad)
u = u_equilibrium(q_ref) − K [q̂−q_ref, q̇̂]ᵀ
Qc = diag(20 I₆, I₆), Rc = 0.1 I₆</pre>
<p>The MuJoCo transition is linearised locally using finite differences, including actual position-actuator dynamics. The model is refreshed every 0.5 simulated seconds around estimated joints. Initial controllability and observability ranks are both 12/12. The initial closed-loop spectral radius is about 0.9911; a small-perturbation one-step residual is about 10⁻⁶ in the mixed state coordinates.</p>
<p>These are local properties. Contact, torque saturation and large posture changes invalidate a global linear-model interpretation. Actuator target clipping is logged separately. In the corrected seed-7 sequence there was no target clipping, but at least one arm actuator reached its force limit in about 0.037% of samples. These are different saturation measures; see the <a href="static/evidence/sequence_seed7_detection_fix/force_limits.json">force-limit analysis</a>. A tiny one-step residual does not establish full-trajectory model accuracy.</p>
<p>Joint KF defaults: R = (0.0005 rad)² I₆ and Q = 10⁻⁵ I₁₂. Joseph covariance updates are used. Equal numerical process variances across position and velocity have different units and are a provisional modelling choice.</p></section>
<section><h2>Camera validation</h2><p>A static bench places one cube at nine known table locations (x = 0.28, 0.32, 0.36 m; y = −0.12, 0, 0.12 m), with five observations each. Robot pose is set explicitly for this sensor bench; it is not a pick-and-place experiment. Localisation uses noisy encoder-derived FK, colour/depth measurements and the known 40 mm cube geometry.</p>
<div class="scroll"><table><tr><th>Revision / experiment</th><th>Added pixel depth noise</th><th>Detected / attempted</th><th>3D RMSE</th><th>Maximum error</th><th>Raw data</th></tr>'''+''.join(camera_rows)+'''</table></div>
<p>The earlier matching-colour geometry detector accepted 40/45 observations (88.9%) and had about 1.49 mm error on accepted observations. The current side-view mounting accepts 45/45 static grid observations, with about 2.48 mm RMSE and 3.49 mm maximum error. Accuracy and detection coverage must be reported together. These are static sensor tests, not moving-grasp robustness trials. Simulator intrinsics and mounting are known. This validates simulated localisation, not physical camera calibration. The surface-to-centre correction is approximate and produces view-dependent bias. Median aggregation reduces independent per-pixel noise; correlated bias, other objects and other camera angles remain untested.</p>
<img src="static/evidence/final_calibration_clean/localisation.png" alt="Localisation errors at known grid points"></section>
<section><h2>Controller and estimator trade-offs</h2><p>Thirty local joint-step experiments: five settings × two encoder-noise levels × three matched seeds. Each run lasts 2.5 simulated seconds and applies a small reference step near HOME. This isolates local feedback behaviour; it does not establish sorting robustness.</p>
<div class="scroll"><table><tr><th>Setting</th><th>Encoder σ (mrad)</th><th>Tracking RMSE (mrad)</th><th>Angle estimate RMSE (mrad)</th><th>Velocity estimate RMSE (mrad/s)</th><th>∫Στ²dt (N²m²s)</th></tr>'''+''.join(control_rows)+'''</table></div>
<p>Increasing the position weight from 20 to 80 reduces local tracking RMSE by about 28%. At nominal encoder noise it costs about 9% more torque-squared effort; at four times the encoder noise, it costs about 61% more. Torque squared is an effort proxy, not mechanical energy. The baseline weight is retained pending integrated comparisons.</p>
<p>Reducing observer Q from 10⁻⁵ I to 10⁻⁷ I reduces nominal angle error from 0.484 to 0.271 mrad but increases velocity error from 1.93 to 7.00 mrad/s. Tuning must therefore consider responsiveness as well as angle smoothing. NIS and marginal uncertainty coverage are recorded in the raw study; no full consistency claim is made.</p>
<img src="static/evidence/local_control_study/tradeoffs.png" alt="Measured control and estimation trade-offs"><p><a href="static/evidence/local_control_study/trials.csv">All 30 local trials</a></p></section>
<section><h2>Continuous moving-target demonstration</h2><p>Run the main script without colours to enter the pick-and-deposit order in the terminal, for example r,b,y. The current matte-grey scene uses a physical platform to move the red cube side to side along Y, +/-20 mm (4 cm total) with a 20 s cycle. Peak commanded speed is about 6.28 mm/s, with smooth natural reversals. It keeps moving through closure and lift. The robot tracks camera-filtered position and velocity during approach, descent and closure; it never receives the platform phase or true cube position as its target.</p><p>The current grey-table side-to-side revision completed full red/blue/yellow and blue/yellow/red sequences on seed 7. Recorded red travel was 40.0 mm before grasp in both runs; pre-contact horizontal speeds were 2.675 and 5.547 mm/s. All six pickups and placements were visually confirmed and independently scored correct. Eleven behaviour tests pass. These selected order tests do not establish all-seed reliability. The speed is an evaluation-only event, not a feedback input. The current fixed camera looks between the fingers. Default XML cube poses are excluded from the separate kinematic/linearisation workspace to avoid fictitious object contacts; all real scene contacts remain active.</p><img src="static/evidence/clear_sweep_rby_seed7/live_tracking.png" alt="Measured moving cube positions and velocities, raw RGB-D versus Kalman estimates and evaluation truth"><p>The plot uses fresh accepted detections, with nominal KF uncertainty. View-dependent centre bias and imperfect velocity estimates remain visible; a successful grasp does not prove optimal estimation. Final descent and closure allow at most 3 s of prediction through occlusion, and pickup confirmation still requires fresh frames. Earlier live-development failures involved partial-view bias, hidden cubes and contact-contaminated scheduled models.</p></section><section><h2>Sensor verification and limitations</h2><p>Pickup requires a measured lift of at least 35 mm, proximity to the tool, stable relative offset over four fresh frames, and supporting finger displacement/force. Missing vision is unconfirmed. Carry checks detect relative visual drift or loss of gripper force. Placement requires an open gripper and four fresh cube observations inside a conservative tray interior and settled height range. Ground-truth scoring is separate from these decisions.</p>
<p>The current default moving-target scenario uses a physical stage with +/-20 mm side-to-side travel and a 20 s period. The stage keeps moving during closure and lift; friction and gripper contacts determine the free cube motion, with no cube-pose override or scripted stop at closure. This is a bounded moving-stage experiment, not an industrial conveyor validation. Expanded random seeds, viewing angles, occlusion, bias disturbances and sensor false-positive/false-negative studies remain necessary.</p></section>
<section><h2>Reproduction</h2><pre>python -m pip install -r requirements.txt
python main_final_integration.py red blue yellow --headless --seed 7 --output results/replay7
python -m unittest discover -s tests -v
python tools/validate_camera.py --output results/camera_new
python tools/evaluate_control.py --output results/control_new
python tools/run_batch.py --seeds 7 11 23 --conditions nominal encoder_x4 depth_5mm --output results/new_batch
python tools/plot_results.py results/replay7</pre>
<p>Run from the repository root. Headless rendering still requires OpenGL. JSON summaries, CSV trajectories, visual measurements and last camera frames preserve evidence. Do not combine trials from different source hashes into one reliability claim. The README documents assumptions and CLI options.</p></section>
<section><h2>Team and AI declaration</h2><p>Tasnim Sarah · Maryam Qutub · Princess Kisakye · Vanessa Kurniawidi. The team must supply accurate individual contributions and a five-minute demo video.</p><p>OpenAI Codex assisted with review, LQR integration, visual verification, experiment tooling and documentation for this revision. Team members must review the changes and understand the complete pipeline. The final declaration must include all other tools used.</p></section>
</main></body></html>'''
    (ROOT/'docs'/'index.html').write_text(document,encoding='utf-8')
    print('Updated docs/index.html with recorded results')


if __name__=='__main__':
    main()
