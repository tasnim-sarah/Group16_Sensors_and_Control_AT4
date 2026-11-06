"""Estimated-state LQR sorting. Truth access is confined to sensors/setup/evaluation."""
import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from cameramodulekal import CameraSensor, camera_pose_from_joint_positions
from controlfile import (PositionActuatorStateSpaceModel, JointKalmanObserver,
                         PositionActuatorLQRController)
from robot_scene import HOME, FINGER_TRAVEL, TRAYS, TRAY_BODY, TRAY_XY, build_scene, platform_target
from visual_verification import PickupVerifier, placement_observation


class TaskFailure(RuntimeError):
    pass


class SortingRobot:
    def __init__(self, args):
        self.args = args
        self.active_cube = None
        self.tracking_enabled = False
        self.announced_detections = set()
        source_root = Path(__file__).resolve().parent
        self.source_hashes = {name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
            for name in ('robot_pipeline.py','robot_scene.py','controlfile.py','cameramodulekal.py','visual_verification.py')}
        self.model, self.data = build_scene(args.seed,moving_red=args.red_amplitude>0)
        self.fk = mujoco.MjData(self.model)
        # Kinematic/model workspace has no external object contacts. Default XML
        # cube poses can otherwise create fictitious contact in a scheduled model.
        # The real scene and camera still include all physical contacts normally.
        for index,colour in enumerate(('red','yellow','blue')):
            adr=int(self.model.jnt_qposadr[self.model.joint(colour+'_cube_joint').id])
            self.fk.qpos[adr:adr+3]=[-2-index,-2,2]
        self.site = self.model.site("tcp").id
        self.finger_adr = int(self.model.jnt_qposadr[self.model.joint("finger_left").id])
        self.rng = np.random.default_rng(args.seed + 1000)
        self.dt = float(self.model.opt.timestep)
        self.camera_steps = max(1, round((1 / 15) / self.dt))
        self.step_index = 0
        self.stage = "initialising"
        self.gripper = 0.0
        self.reference = HOME.copy()
        self.previous_control = HOME.copy()
        self.q_est = self.read_encoder()
        self.dq_est = np.zeros(6)
        self.position, self.rotation = self.pose(HOME)
        self.home_position = self.position.copy()
        self.home_rotation = self.rotation.copy()
        self.control_model = PositionActuatorStateSpaceModel(self.model, self.data)
        self.initial_report = self.control_model.analyse()
        if args.debug:
            self.control_model.print_report()
        self.initial_matrices = {key:getattr(self.control_model,key).copy() for key in ('A','B','C','D','offset')}
        self.controller = self.make_controller()
        self.observer = JointKalmanObserver(self.control_model,
            position_noise=args.encoder_noise**2, process_noise=args.observer_q)
        self.observer.reset(self.q_est)
        self.last_schedule = 0.0
        self.radius = float(max(abs(np.linalg.eigvals(
            self.control_model.A - self.control_model.B @ self.controller.K))))
        self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        self.sensor = CameraSensor(self.model, self.renderer, min_area=45,
            depth_noise_std=args.depth_noise, depth_noise_seed=args.seed + 2000,
            workspace_bounds=[[-.35,.65],[-.5,.65],[-.04,.65]], classify_geometry=True)
        # Filter dt must be simulation time between captures, not wall-clock render time.
        for filt in self.sensor.position_filters.values():
            filt.dt = self.camera_steps * self.dt
        self.tracks, self.last_seen, self.raw, self.detections = {}, {}, {}, {}
        self.samples, self.vision_samples, self.events = [], [], []
        self.tracking_samples = []
        self.evaluated_contacts = set()
        self.last_tracking_print = -np.inf
        self.pickup_verifier = None
        self.verification_cube = None
        self.pickup_status = "unconfirmed"
        self.sensor_pickups = self.sensor_placements = 0
        self.completed = 0
        self.carry_offset = None
        self.sensor_samples = []
        self.viewer = None
        self.closed = False
        self.motion_target = None
        self.motion_rotation = self.home_rotation.copy()
        self.next_frame = 0
        self.red_amplitude = args.red_amplitude
        self.red_driven = self.red_amplitude > 0
        self.platform_actuators = ([self.model.actuator('platform_'+axis).id for axis in ('x','y')]
                                   if self.red_driven else [])
        self.live_target_active=self.red_driven
        if not args.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def announce(self, message, debug=False):
        if debug and not self.args.debug:
            return
        tag=self.active_cube.split()[0].upper() if self.active_cube else 'SYSTEM'
        print(f'[{tag}] {message}',flush=True)

    def make_controller(self):
        return PositionActuatorLQRController(self.control_model,
            position_weight=self.args.lqr_position, velocity_weight=self.args.lqr_velocity,
            input_weight=self.args.lqr_input)

    def read_encoder(self):
        # SENSOR BOUNDARY: simulated encoder; truth is never returned separately.
        return self.data.qpos[:6].copy() + self.rng.normal(0, self.args.encoder_noise, 6)

    def finger_holding(self):
        # SENSOR BOUNDARY: finger displacement and actuator force transducer.
        opening = float(self.data.qpos[self.finger_adr])
        force = abs(float(self.data.actuator_force[6]))
        return 0.006 < opening < FINGER_TRAVEL - 0.004 and force > 3.0

    def pose(self, q):
        self.fk.qpos[:6] = q
        self.fk.qvel[:6] = 0
        mujoco.mj_forward(self.model, self.fk)
        return (self.fk.site_xpos[self.site].copy(),
                self.fk.site_xmat[self.site].reshape(3, 3).copy())

    @staticmethod
    def rotation_error(wanted, actual):
        return 0.5 * sum(np.cross(actual[:, i], wanted[:, i]) for i in range(3))

    def ik(self, point, rotation, seed):
        """Kinematic reference generator; does not send actuator commands."""
        q = seed.copy()
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(100):
            p, r = self.pose(q)
            orientation_angle = np.arccos(np.clip((np.trace(rotation @ r.T)-1)/2, -1, 1))
            if np.linalg.norm(point-p) < 0.002 and orientation_angle < 0.025:
                return q, True
            mujoco.mj_jacSite(self.model, self.fk, jp, jr, self.site)
            j = np.vstack([jp[:, :6], 0.3 * jr[:, :6]])
            error = np.r_[point-p, 0.3*self.rotation_error(rotation, r)]
            delta = j.T @ np.linalg.solve(j @ j.T + 0.006**2*np.eye(6), error)
            delta *= min(1.0, 0.12 / max(np.linalg.norm(delta), 1e-12))
            q = np.clip(q + delta, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])
        return q, False

    def schedule(self):
        """Linearise around estimated pose with model-based gravity equilibrium.

        Scheduling improves validity away from HOME. No true joint state is used.
        The resulting controller remains local LQR, subject to saturation.
        """
        self.fk.qpos[self.finger_adr] = self.data.qpos[self.finger_adr]  # finger encoder
        right_adr = int(self.model.jnt_qposadr[self.model.joint('finger_right').id])
        self.fk.qpos[right_adr] = self.fk.qpos[self.finger_adr]
        self.pose(self.q_est)
        self.fk.ctrl[:6] = self.q_est + self.fk.qfrc_bias[:6] / self.model.actuator_gainprm[:6, 0]
        model = PositionActuatorStateSpaceModel(self.model, self.fk)
        self.control_model = model
        self.controller = self.make_controller()
        self.observer.A, self.observer.B = model.A.copy(), model.B.copy()
        self.observer.offset = model.offset.copy()
        self.last_schedule = float(self.data.time)

    def capture(self):
        camera_pose = camera_pose_from_joint_positions(self.model, self.fk, self.q_est)
        rgb, depth = self.sensor.capture(self.data)
        self.last_rgb = rgb
        found = self.sensor.detect(self.data, rgb, depth, camera_pose=camera_pose,
            full_cube_labels=('Red Cube',) if self.live_target_active else ())
        self.raw = {}
        self.sensor_samples.append([float(self.data.time),self.stage,
            float(self.data.qpos[self.finger_adr]),float(self.data.actuator_force[6])])
        for label, detection in found.items():
            raw = detection["position_world_raw_m"].copy()
            filtered = detection["position_world_m"].copy()
            if label in TRAYS:
                # Known cube geometry corrects visible-surface depth to approximate centre.
                for p in (raw, filtered):
                    ray = p - camera_pose[0]
                    p += 0.02 * ray / max(np.linalg.norm(ray), 1e-12)
            if not (-0.35 < raw[0] < 0.65 and -0.5 < raw[1] < 0.65 and -0.04 < raw[2] < 0.65):
                continue
            self.raw[label], self.tracks[label] = raw, filtered
            self.last_seen[label] = float(self.data.time)
            # EVALUATION ONLY: localisation error; never returned to task logic.
            body = label.lower().replace(" ", "_") if label in TRAYS else TRAY_BODY[label]
            truth = self.data.xpos[self.model.body(body).id]
            self.vision_samples.append([float(self.data.time), self.stage, label,
                *raw, *truth.copy(), float(np.linalg.norm(raw-truth))])
            filt=self.sensor.position_filters[label].kf
            estimated_velocity=filt.statePost[3:6,0].astype(float)
            sigma=np.sqrt(np.maximum(np.diag(filt.errorCovPost)[:3],0))
            truth_velocity=np.zeros(6)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,
                self.model.body(body).id,truth_velocity,0)  # EVALUATION ONLY
            self.tracking_samples.append([float(self.data.time),self.stage,label,
                *raw,*filtered,*estimated_velocity,*sigma,*truth.copy(),*truth_velocity[3:]])
        self.detections = found
        if (self.args.debug and self.tracking_enabled and self.active_cube
                and self.data.time-self.last_tracking_print>=1.0):
            self.last_tracking_print=float(self.data.time)
            label=self.active_cube
            if label in self.raw:
                velocity=self.sensor.position_filters[label].kf.statePost[3:6,0]
                self.announce(f"sim time {self.data.time:.1f}s | measured XY {np.round(self.raw[label][:2],4)} m | "
                    f"KF XY {np.round(self.tracks[label][:2],4)} m | velocity {np.round(1000*velocity[:2],2)} mm/s",debug=True)
            elif label in self.last_seen:
                self.announce(f"not visible | last accepted observation {self.data.time-self.last_seen[label]:.2f}s ago",debug=True)
        if self.pickup_verifier is not None:
            p, r = self.pose(self.q_est)
            self.pickup_status = self.pickup_verifier.observe(
                self.raw.get(self.verification_cube), p, r, self.finger_holding())
        if self.carry_offset is not None:
            if not self.finger_holding():
                raise TaskFailure('carry_gripper_sensor_lost:'+self.verification_cube)
            if self.verification_cube in self.raw:
                p,r = self.pose(self.q_est)
                offset = r.T @ (self.raw[self.verification_cube]-p)
                if np.linalg.norm(offset-self.carry_offset)>.04:
                    raise TaskFailure('carry_visual_slip:'+self.verification_cube)
        if self.motion_target is not None:
            p, _ = self.pose(self.q_est)
            point = np.asarray(self.motion_target())
            delta = point-p
            waypoint = p + delta * min(1.0, 0.012/max(np.linalg.norm(delta), 1e-12))
            solution, ok = self.ik(waypoint, self.motion_rotation, self.q_est)
            if ok:
                self.reference = solution
        if not self.args.headless:
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for label, detection in found.items():
                u, v = map(int, detection["pixel"])
                cv2.putText(display, label, (u, v), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.putText(display, self.stage, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            cv2.putText(display, "LQR + KF | pickup: " + self.pickup_status, (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
            label=self.active_cube
            if self.tracking_enabled and label in self.raw:
                v=self.sensor.position_filters[label].kf.statePost[3:6,0]
                cv2.putText(display,f"{label}: measured XY {np.round(self.raw[label][:2],3)} | KF {np.round(self.tracks[label][:2],3)} m",
                    (8,64),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                cv2.putText(display,f"Estimated velocity XY: {1000*v[0]:+.1f}, {1000*v[1]:+.1f} mm/s",
                    (8,84),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
            elif self.tracking_enabled and label in self.last_seen:
                text=f'{label}: not visible'
                if label=='Red Cube' and self.live_target_active:
                    text+=' - temporary prediction'
                if self.args.debug:
                    text+=f' | last observation {self.data.time-self.last_seen[label]:.2f}s ago'
                cv2.putText(display,text,
                    (8,64),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1)
            cv2.imshow("Wrist RGB-D verification", display)
            if cv2.waitKey(1) & 0xff == ord('q'):
                raise TaskFailure("user_stopped")
            self.viewer.sync()

    def step(self):
        if self.viewer is not None and not self.viewer.is_running():
            raise TaskFailure("viewer_closed")
        if self.red_driven:
            # ENVIRONMENT ONLY. Never overwrite cube qpos/qvel or disable motion
            # at closure. Platform joint velocity transfers motion through friction.
            self.data.ctrl[self.platform_actuators]=platform_target(
                self.data.time,self.red_amplitude,self.args.red_period)
        if self.step_index:
            self.q_est, self.dq_est = self.observer.update(self.read_encoder(), self.previous_control)
        if self.data.time - self.last_schedule >= 0.5:
            self.schedule()
        if self.step_index % self.camera_steps == 0:
            self.capture()
        raw = self.controller.raw_command(self.reference, self.q_est, self.dq_est)
        command = np.clip(raw, self.control_model.ctrl_limits[:, 0], self.control_model.ctrl_limits[:, 1])
        self.data.ctrl[:6], self.data.ctrl[6] = command, self.gripper
        self.previous_control = command.copy()
        # EVALUATION ONLY: neither these arrays nor their errors enter feedback.
        truth_q, truth_dq = self.data.qpos[:6].copy(), self.data.qvel[:6].copy()
        self.samples.append([float(self.data.time), self.stage, *self.reference, *self.q_est,
            *self.dq_est, *truth_q, *truth_dq, *command,
            int(np.any(abs(raw-command)>1e-10)), *self.observer.position_standard_deviation(),
            getattr(self.observer, 'last_nis', 0.0), *self.data.actuator_force[:6].copy()])
        # EVALUATION ONLY: retain pre-contact velocity to verify the moving-grasp
        # experiment. This value is never supplied to tracking or task decisions.
        contact_cube=self.verification_cube if self.gripper==FINGER_TRAVEL else None
        pre_contact_velocity=None
        if contact_cube and contact_cube not in self.evaluated_contacts:
            colour=contact_cube.split()[0].lower()
            dof=int(self.model.jnt_dofadr[self.model.joint(colour+'_cube_joint').id])
            pre_contact_velocity=self.data.qvel[dof:dof+3].copy()
        mujoco.mj_step(self.model, self.data)
        if pre_contact_velocity is not None:
            cube_geom=self.model.geom(colour+'_cube_geom').id
            fingers={self.model.geom('finger_'+side+'_pad').id for side in ('left','right')}
            touching=any((int(c.geom1)==cube_geom and int(c.geom2) in fingers)
                         or (int(c.geom2)==cube_geom and int(c.geom1) in fingers)
                         for c in self.data.contact)
            if touching:
                self.evaluated_contacts.add(contact_cube)
                speed=float(np.linalg.norm(pre_contact_velocity[:2]))
                self.events.append({'cube':contact_cube,'event':'first_finger_contact_evaluation',
                    'time_s':float(self.data.time),'pre_contact_velocity_m_s':pre_contact_velocity.tolist(),
                    'pre_contact_horizontal_speed_m_s':speed,'platform_motion_enabled':self.red_driven})
                self.announce(f'Evaluation only: pre-contact {contact_cube} speed = {speed*1000:.3f} mm/s',debug=True)
        self.step_index += 1

    def hold(self, seconds):
        end = self.data.time + seconds
        while self.data.time < end:
            self.step()

    def move(self, point, stage, rotation=None, tolerance=0.009, timeout=18.0):
        self.stage = stage
        self.announce(stage,debug=True)
        self.motion_target = point if callable(point) else lambda: np.asarray(point)
        self.motion_rotation = self.home_rotation.copy() if rotation is None else rotation.copy()
        start, settled = float(self.data.time), 0
        try:
            while self.data.time-start < timeout:
                self.step()
                p, r = self.pose(self.q_est)
                pe = np.linalg.norm(self.motion_target()-p)
                re = np.arccos(np.clip((np.trace(self.motion_rotation @ r.T)-1)/2, -1, 1))
                settled = settled+1 if pe < tolerance and re < 0.07 else 0
                if settled*self.dt >= 0.10:
                    return
            raise TaskFailure(f"motion_timeout:{stage}:position_error={pe:.4f}:rotation_error={re:.4f}")
        finally:
            self.motion_target = None

    def acquire(self, label, timeout=2.0):
        start = self.data.time
        while self.data.time-start < timeout:
            self.step()
            if label in self.raw and self.data.time-self.last_seen[label] < 0.10:
                if label not in self.announced_detections:
                    self.announced_detections.add(label)
                    if label.endswith('Cube'):
                        self.announce('Cube detected - live tracking active.')
                    else:
                        self.announce('Matching tray detected.')
                return self.tracks[label].copy()
        raise TaskFailure("target_not_visible:"+label)

    def aim_rotation(self, point):
        cam_pos, cam_rot = camera_pose_from_joint_positions(self.model, self.fk, self.q_est)
        _, tcp_rot = self.pose(self.q_est)
        mount = tcp_rot.T @ cam_rot
        z = cam_pos-np.asarray(point)
        z /= np.linalg.norm(z)
        ref = np.array([0.,0.,1.]) if abs(z[2])<.98 else np.array([0.,1.,0.])
        x = np.cross(ref,z); x /= np.linalg.norm(x)
        desired_cam = np.column_stack([x,np.cross(z,x),z])
        return desired_cam @ mount.T

    def locate_tray(self, tray):
        # Known workspace prior only directs a camera search. Never used as drop target.
        prior = np.r_[TRAY_XY[tray], 0.02]
        self.announce('Finding the matching tray before pickup (camera search).')
        self.move(np.array([.34, 0., .25]), "Camera search: clear workspace")
        self.move(np.r_[prior[:2], .26], "Camera search: find "+tray)
        measured = self.acquire(tray)
        self.announce(f'Destination measurement: {tray} {np.round(measured,3)} m',debug=True)
        self.move(np.array([.32, 0., .25]), "Camera search complete: return to cubes")
        return measured

    def pick_place(self, cube):
        self.active_cube=cube
        self.tracking_enabled=False
        self.pickup_status='unconfirmed'
        self.announced_detections.clear()
        self.last_tracking_print=-np.inf
        tray = TRAYS[cube]
        measured_tray = self.locate_tray(tray)
        self.announce('Searching for cube.')
        self.acquire(cube)
        self.tracking_enabled=True
        def tracked_target(height):
            age=float(self.data.time-self.last_seen.get(cube, -np.inf))
            allowed_age=3.0 if cube=='Red Cube' and self.live_target_active and height<.05 else .7
            if age>allowed_age:
                raise TaskFailure("stale_cube_target:"+cube)
            position=self.tracks[cube].copy()
            if cube=='Red Cube' and self.red_driven:
                velocity=self.sensor.position_filters[cube].kf.statePost[3:6,0].astype(float)
                # This prediction knows no platform phase, period or true cube pose.
                velocity=np.clip(velocity,-.025,.025)
                age=float(self.data.time-self.last_seen[cube])
                position+=velocity*(age+.18)
            return np.r_[position[:2],height]
        moving_target=cube=='Red Cube' and self.red_driven
        self.announce('Approaching cube - '+('following live motion.' if moving_target else 'preparing pickup.'))
        self.move(lambda:tracked_target(.14), "Approach "+cube)
        self.hold(.25)
        spot = self.acquire(cube)
        grasp_height=.033 if moving_target else .030
        descent=(lambda:tracked_target(grasp_height)) if moving_target else np.r_[spot[:2],grasp_height]
        self.move(descent, "Live-tracked descent to "+cube if moving_target else "Descend to "+cube,tolerance=.005)
        initial = self.raw.get(cube, spot).copy()
        self.pickup_verifier = PickupVerifier(initial)
        self.verification_cube = cube
        self.gripper = FINGER_TRAVEL
        self.stage = "Closing gripper"
        self.announce('Closing gripper and checking pickup.')
        if moving_target:
            self.motion_target=lambda:tracked_target(grasp_height)
            self.motion_rotation=self.home_rotation.copy()
        try:
            self.hold(.7)
        finally:
            self.motion_target=None
        if moving_target:
            self.live_target_active=False  # Lift verification accepts partial faces again.
        p, _ = self.pose(self.q_est)
        self.move(np.r_[p[:2], .16], "Verification lift "+cube, tolerance=.008)
        self.hold(.5)
        self.events.append({"cube":cube,"event":"pickup","status":self.pickup_status,
                            "gripper_holding":bool(self.finger_holding())})
        if self.pickup_status != "confirmed":
            raise TaskFailure("pickup_"+self.pickup_status+":"+cube)
        self.announce('Pickup confirmed.')
        self.sensor_pickups += 1
        self.carry_offset = np.mean(self.pickup_verifier.offsets,axis=0)
        self.pickup_verifier = None
        self.announce('Transferring to matching tray.')
        self.move(np.r_[measured_tray[:2], .22], "Transfer "+cube)
        if not self.finger_holding():
            raise TaskFailure("gripper_lost_cube:"+cube)
        self.move(np.r_[measured_tray[:2], .11], "Release approach "+cube)
        self.carry_offset = None
        self.gripper = 0
        self.stage = "Releasing "+cube
        self.hold(.8)
        p, _ = self.pose(self.q_est)
        self.move(np.r_[p[:2], .24], "Clear view for placement")
        p, _ = self.pose(self.q_est)
        self.move(p, "Inspect placement")
        consecutive, status = 0, "unconfirmed"
        end = self.data.time+2
        while self.data.time < end:
            self.step()
            if self.step_index % self.camera_steps != 1:
                continue
            status = placement_observation(self.raw.get(cube), measured_tray,
                self.data.qpos[self.finger_adr] < .004)  # finger encoder
            consecutive = consecutive+1 if status == "confirmed" else 0
            if consecutive >= 4:
                break
        self.events.append({"cube":cube,"event":"placement","status":status})
        if consecutive < 4:
            raise TaskFailure("placement_"+status+":"+cube)
        self.sensor_placements += 1
        self.completed += 1
        self.tracking_enabled=False
        self.announce('Placement confirmed - tracking finished.')
        self.announce('Returning home before the next cube.')
        self.move(np.array([.34, 0., .25]), "Return through clear workspace")
        self.move(self.home_position, "Return home")
        self.announce('Task complete.')
        self.active_cube=None

    def save(self, failure, elapsed):
        directory = Path(self.args.output)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory/'initial_model.npz',**self.initial_matrices)
        if hasattr(self, 'last_rgb'):
            cv2.imwrite(str(directory/'last_camera.png'), cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR))
        def write_csv(name, headers, rows):
            with (directory/name).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle); writer.writerow(headers); writer.writerows(rows)
        vector_names = ['reference','estimate_q','estimate_dq','truth_q','truth_dq','command']
        columns = ['time_s','stage'] + [f'{name}_{i+1}' for name in vector_names for i in range(6)]
        columns += ['saturated'] + [f'sigma_q_{i+1}' for i in range(6)] + ['nis']
        columns += [f'torque_{i+1}' for i in range(6)]
        write_csv('trajectory.csv', columns, self.samples)
        write_csv('vision.csv',['time_s','stage','label','x','y','z','truth_x','truth_y','truth_z','error_m'],self.vision_samples)
        write_csv('gripper.csv',['time_s','stage','finger_displacement_m','finger_force_n'],self.sensor_samples)
        write_csv('object_tracking.csv',['time_s','stage','label']+
            [name+'_'+axis for name in ('raw','filtered','velocity','sigma','truth_position','truth_velocity')
             for axis in ('x','y','z')],self.tracking_samples)
        # EVALUATION ONLY: independent physical placement scoring, never task control.
        truth_placements = {}
        for cube in self.args.order:
            p = self.data.xpos[self.model.body(cube.lower().replace(' ','_')).id]
            t = self.data.xpos[self.model.body(TRAY_BODY[TRAYS[cube]]).id]
            truth_placements[cube] = bool(np.all(abs(p[:2]-t[:2])<=.027) and .008<=p[2]<=.055)
        a = np.asarray([row[2:] for row in self.samples], dtype=float)
        result = {'timestamp':datetime.now().astimezone().isoformat(), 'config':vars(self.args),
            'source_sha256':self.source_hashes,
            'run_type':'control_smoke' if self.args.smoke_seconds else 'sorting',
            'sequence_success':failure is None and self.completed==len(self.args.order),
            'failure':failure,'completed':self.completed,'pickup_confirmations':self.sensor_pickups,
            'placement_confirmations':self.sensor_placements,'truth_placements':truth_placements,
            'simulation_time_s':float(self.data.time),'wall_time_s':elapsed,'events':self.events,
            'controllability_rank':self.initial_report['controllability_rank'],
            'observability_rank':self.initial_report['observability_rank'],
            'initial_closed_loop_spectral_radius':self.radius,
            'local_model_validation_max_error':self.control_model.validation_max_error}
        if len(a):
            result.update(joint_tracking_rmse_rad=float(np.sqrt(np.mean((a[:,:6]-a[:,18:24])**2))),
                joint_estimation_rmse_rad=float(np.sqrt(np.mean((a[:,6:12]-a[:,18:24])**2))),
                velocity_estimation_rmse_rad_s=float(np.sqrt(np.mean((a[:,12:18]-a[:,24:30])**2))),
                saturation_fraction=float(np.mean(a[:,36])),
                torque_squared_integral=float(np.sum(a[:,44:50]**2)*self.dt))
        (directory/'summary.json').write_text(json.dumps(result, indent=2),encoding='utf-8')
        if self.args.debug:
            print(json.dumps(result, indent=2), flush=True)
        else:
            print(f'[SYSTEM] {self.completed}/{len(self.args.order)} requested cubes deposited and confirmed.',flush=True)
            print(f'[SYSTEM] Detailed results saved to {directory / "summary.json"}',flush=True)

    def close(self):
        self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()
            cv2.destroyAllWindows()


def parse_pick_order(text):
    names={token:colour+' Cube' for colour in ('Red','Blue','Yellow')
           for token in (colour.lower(),colour[0].lower())}
    tokens=[token for token in re.split(r'[,\s]+',text.strip().lower()) if token]
    if not tokens:
        raise ValueError('Enter at least one colour, for example r,b,y.')
    unknown=[token for token in tokens if token not in names]
    if unknown:
        raise ValueError('Unknown colour(s): '+', '.join(unknown)+'. Use r, b, y or full colour names.')
    order=[names[token] for token in tokens]
    if len(order)!=len(set(order)):
        raise ValueError('Choose each cube at most once.')
    return order


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('colours', nargs='*',help='Optional order; otherwise interactive runs prompt for it')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--debug',action='store_true',help='Print model, coordinates, filter diagnostics and full result JSON')
    parser.add_argument('--output', default='results/latest')
    parser.add_argument('--encoder-noise', type=float, default=.0005)
    parser.add_argument('--depth-noise', type=float, default=0)
    parser.add_argument('--observer-q', type=float, default=1e-5)
    parser.add_argument('--lqr-position', type=float, default=20)
    parser.add_argument('--lqr-velocity', type=float, default=1)
    parser.add_argument('--lqr-input', type=float, default=.1)
    parser.add_argument('--red-amplitude', type=float, default=.020,
                        help='Side-to-side half-travel in metres (default .020 = 4 cm total; 0 disables motion)')
    parser.add_argument('--red-period',type=float,default=20,
                        help='Full side-to-side cycle in seconds (default 20)')
    parser.add_argument('--smoke-seconds', type=float, default=0,
                        help='Hold HOME for this many seconds instead of sorting')
    args = parser.parse_args(argv)
    for key in ('encoder_noise','depth_noise','observer_q','red_amplitude','smoke_seconds'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<0:
            parser.error(key+' must be finite and nonnegative')
    for key in ('lqr_position','lqr_velocity','lqr_input','red_period'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0:
            parser.error(key+' must be finite and positive')
    if args.red_amplitude>.05:
        parser.error('--red-amplitude must not exceed .05 m for this platform/workspace')
    if not args.colours and not args.headless and not args.smoke_seconds:
        while True:
            try:
                text=input('Pick and deposit order (e.g. r,b,y or y r b): ')
                args.order=parse_pick_order(text)
                break
            except ValueError as error:
                print(error)
            except EOFError:
                parser.error('No terminal input available. Supply the order, e.g. exampletestfile.py r,b,y')
    else:
        try:
            args.order=parse_pick_order(' '.join(args.colours) if args.colours else 'r,b,y')
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    print('Requested pick AND deposit order: '+' -> '.join(args.order),flush=True)
    robot = SortingRobot(args)
    failure = None
    start = time.perf_counter()
    try:
        robot.stage = 'Settling at HOME'
        robot.hold(max(.5,args.smoke_seconds))
        if not args.smoke_seconds:
            for cube in args.order:
                robot.pick_place(cube)
    except TaskFailure as error:
        failure = str(error)
        print('STOPPED:',failure,flush=True)
    except BaseException as error:
        failure = f'{type(error).__name__}:{error}'
        raise
    finally:
        try:
            robot.save(failure, time.perf_counter()-start)
        finally:
            robot.close()
    return int(failure is not None)
