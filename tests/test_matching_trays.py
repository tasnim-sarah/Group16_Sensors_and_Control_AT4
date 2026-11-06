"""RGB-D regression: distinguish a cube from its same-colour tray, without IDs."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
import mujoco
import numpy as np
from robot_pipeline import SortingRobot,parse_args
from robot_scene import TRAYS,TRAY_XY


class MatchingTrayTests(unittest.TestCase):
    def test_diagnostics_follow_active_target_and_stop_when_tracking_finishes(self):
        robot=SortingRobot(parse_args(['--headless']))
        try:
            q,ok=robot.ik(np.array([.34,0,.25]),robot.home_rotation,robot.q_est)
            self.assertTrue(ok)
            robot.data.qpos[:6]=q;robot.q_est=q
            mujoco.mj_forward(robot.model,robot.data)
            robot.active_cube='Yellow Cube';robot.tracking_enabled=True
            robot.args.debug=True
            output=StringIO()
            with redirect_stdout(output):robot.capture()
            self.assertIn('[YELLOW]',output.getvalue())
            self.assertNotIn('[RED]',output.getvalue())
            self.assertNotIn('LIVE RED',output.getvalue())
            robot.tracking_enabled=False;robot.data.time+=2
            output=StringIO()
            with redirect_stdout(output):robot.capture()
            self.assertEqual(output.getvalue(),'')
            robot.args.debug=False;robot.tracking_enabled=True;robot.data.time+=2
            with redirect_stdout(output):robot.capture()
            self.assertEqual(output.getvalue(),'')
        finally:
            robot.close()

    def test_empty_rim_and_cube_inside_for_each_colour(self):
        robot=SortingRobot(parse_args(['--headless']))
        try:
            scan,ok=robot.ik(np.array([.34,0,.25]),robot.home_rotation,robot.q_est)
            self.assertTrue(ok)
            for cube,tray in TRAYS.items():
                with self.subTest(cube=cube):
                    xy=np.asarray(TRAY_XY[tray])
                    q,ok=robot.ik(np.r_[xy,.24],robot.home_rotation,scan)
                    self.assertTrue(ok)
                    robot.q_est=q;robot.data.qpos[:6]=q
                    colour=cube.split()[0].lower()
                    adr=int(robot.model.jnt_qposadr[robot.model.joint(colour+'_cube_joint').id])
                    # Scene truth is used only to arrange these test fixtures.
                    robot.data.qpos[adr:adr+3]=[-.6,-.6,.02]
                    mujoco.mj_forward(robot.model,robot.data);robot.capture()
                    self.assertIn(tray,robot.raw)
                    self.assertNotIn(cube,robot.raw,'Empty tray must not be reported as a cube')
                    robot.data.qpos[adr:adr+3]=[*xy,.025]
                    robot.data.time+=robot.camera_steps*robot.dt
                    mujoco.mj_forward(robot.model,robot.data);robot.capture()
                    self.assertIn(cube,robot.raw)
                    self.assertIn(tray,robot.raw)
                    self.assertLess(np.linalg.norm(robot.raw[cube][:2]-xy),.015)
        finally:
            robot.close()


if __name__=='__main__':
    unittest.main()
