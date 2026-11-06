"""Input ordering and a physically transported moving target."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from main_final_integration import parse_args,parse_pick_order
from robot_scene import build_scene,platform_target


class LiveMotionTests(unittest.TestCase):
    def test_order_preserves_requested_sequence(self):
        self.assertEqual(parse_pick_order('r,b,y'),['Red Cube','Blue Cube','Yellow Cube'])
        self.assertEqual(parse_pick_order(' Y, red  b '),['Yellow Cube','Red Cube','Blue Cube'])
        for bad in ('','r,r','r,g'):
            with self.assertRaises(ValueError):parse_pick_order(bad)

    def test_terminal_prompts_and_retries_invalid_order(self):
        with patch('builtins.input',side_effect=['r,r','b,y,r']) as prompt:
            args=parse_args([])
        self.assertEqual(prompt.call_count,2)
        self.assertEqual(args.order,['Blue Cube','Yellow Cube','Red Cube'])
        self.assertGreater(args.red_amplitude,0)
        self.assertFalse(args.debug)

    def test_explicit_or_headless_order_never_prompts(self):
        with patch('builtins.input',side_effect=AssertionError('Unexpected prompt')):
            self.assertEqual(parse_args(['y,r,b']).order,['Yellow Cube','Red Cube','Blue Cube'])
            self.assertEqual(parse_args(['--headless']).order,['Red Cube','Blue Cube','Yellow Cube'])

    def test_platform_contact_physically_moves_free_cube(self):
        model,data=build_scene(7,moving_red=True)
        joint=model.joint('red_cube_joint').id
        adr=int(model.jnt_qposadr[joint]);dof=int(model.jnt_dofadr[joint])
        start=data.qpos[adr:adr+3].copy()
        actuators=[model.actuator('platform_'+axis).id for axis in ('x','y')]
        for _ in range(1000):
            data.ctrl[actuators]=platform_target(data.time,.020,20)
            mujoco.mj_step(model,data)
        self.assertGreater(data.qpos[adr+1]-start[1],.008)
        self.assertLess(abs(data.qpos[adr]-start[0]),.001)
        self.assertGreater(np.linalg.norm(data.qvel[dof:dof+2]),.001)
        self.assertLess(np.linalg.norm(data.qvel[dof:dof+2]),.008)
        self.assertLess(abs(data.qpos[adr+2]-start[2]),.001)

    def test_side_to_side_path_has_four_centimetre_span_without_x_travel(self):
        path=np.array([platform_target(t,.020,20) for t in np.linspace(0,20,101)])
        self.assertAlmostEqual(float(np.ptp(path[:,0])),0)
        self.assertAlmostEqual(float(np.ptp(path[:,1])),.040)


if __name__=='__main__':
    unittest.main()
