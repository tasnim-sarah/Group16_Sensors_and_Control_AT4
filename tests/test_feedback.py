"""Behaviour checks for real actuator dynamics and sensor-only decisions."""
import unittest
import mujoco
import numpy as np
from types import SimpleNamespace
from cameramodulekal import CameraSensor
from controlfile import PositionActuatorStateSpaceModel, PositionActuatorLQRController, JointKalmanObserver
from robot_scene import build_scene, HOME
from visual_verification import PickupVerifier, placement_observation


class FeedbackTests(unittest.TestCase):
    def test_floor_colour_does_not_hide_valid_cube(self):
        model=SimpleNamespace(camera=lambda name:SimpleNamespace(id=0),cam_fovy=[75])
        renderer=SimpleNamespace(width=320,height=240)
        sensor=CameraSensor(model,renderer,min_area=45,
            workspace_bounds=[[-.5,.5],[-.5,.5],[-.04,.5]])
        rgb=np.zeros((240,320,3),dtype=np.uint8)
        depth=np.ones((240,320),dtype=np.float32)
        rgb[20:80,20:80]=[30,90,230]  # Larger blue background patch below table.
        rgb[110:130,150:170]=[30,90,230]
        depth[110:130,150:170]=.2
        found=sensor.detect(SimpleNamespace(time=0),rgb,depth,
                            camera_pose=(np.array([0,0,.2]),np.eye(3)))
        self.assertIn('Blue Cube',found)
        self.assertAlmostEqual(found['Blue Cube']['position_world_raw_m'][2],0,places=6)

    def test_lqr_tracks_small_step_using_noisy_encoder_observer(self):
        model, data = build_scene(7)
        dynamics = PositionActuatorStateSpaceModel(model, data)
        controller = PositionActuatorLQRController(dynamics)
        observer = JointKalmanObserver(dynamics)
        rng = np.random.default_rng(33)
        q, dq = observer.reset(HOME)
        target = HOME + np.array([.025, 0, 0, 0, 0, 0])
        command = HOME.copy()
        errors = []
        for _ in range(750):
            q, dq = observer.update(data.qpos[:6]+rng.normal(0,.0005,6),command)
            command = controller.command(target,q,dq)
            data.ctrl[:6] = command
            mujoco.mj_step(model,data)
            errors.append(np.linalg.norm(data.qpos[:6]-target))
        self.assertLess(np.mean(errors[-100:]), .004)
        self.assertTrue(dynamics.analyse()['controllable'])
        self.assertTrue(dynamics.analyse()['observable'])
        self.assertLess(np.max(abs(np.linalg.eigvals(dynamics.A-dynamics.B@controller.K))),1)
        self.assertGreater(np.min(np.linalg.eigvalsh(observer.P)), -1e-10)

    def test_visual_pickup_requires_lift_and_fresh_frames(self):
        verifier = PickupVerifier([.3,0,.02])
        for _ in range(6):
            self.assertNotEqual(verifier.observe([.3,0,.02],[.3,0,.15],np.eye(3),True),'confirmed')
        for _ in range(4):
            status = verifier.observe([.3,0,.13],[.3,0,.15],np.eye(3),True)
        self.assertEqual(status,'confirmed')
        self.assertEqual(verifier.observe(None,[.3,0,.15],np.eye(3),True),'unconfirmed')
        self.assertEqual(verifier.observe([.3,0,.13],[.3,0,.15],np.eye(3),False),'failed')

    def test_visual_placement_rejects_missing_or_outside_measurement(self):
        self.assertEqual(placement_observation(None,[.3,.2,.02],True),'unconfirmed')
        self.assertEqual(placement_observation([.38,.2,.02],[.3,.2,.02],True),'failed')
        self.assertEqual(placement_observation([.3,.2,.027],[.3,.2,.02],True),'confirmed')


if __name__ == '__main__':
    unittest.main()
