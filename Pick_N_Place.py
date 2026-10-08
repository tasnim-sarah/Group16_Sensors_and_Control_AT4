import os 
import swift
import time
import numpy as np
import roboticstoolbox as rtb
from math import pi, sqrt
from ir_support.robots.UTSMeshRobot import UTSMeshRobot
from ir_support_extra_robots.robots import Turtlebot3Waffle
from roboticstoolbox import DHRobot, DHLink, jtraj, ctraj, models
from ir_support import CylindricalDHRobotPlot
from ir_support_extra_parts.parts import part_names, part_mesh
from ir_support.robots import UR3e
from spatialmath.base import *
from spatialmath import SE3, SO3
from spatialgeometry import Sphere, Arrow, Mesh, Cuboid
           
## SWIFT ENVIROMENT
env = swift.Swift()
env.launch(realtime=True)
env.set_camera_pose([4, 3.5, 1.25], [2.78, 2, 0.47])  # Set camera pose (position, look-at)

## MESHES
# Table
simpleTable = part_mesh("SimpleTable")
simpleTable.T = SE3(2.15, 2, 0)
simpleTable_location = simpleTable.T
simpleTable.scale = (1.5, 1.75, 1)
env.add(simpleTable)

# ROBOTS
# UR3e
UR3e_robot = UR3e()
UR3e_robotx = 1.5
UR3e_roboty = 2
UR3e_robotz = 0.62
UR3e_robot.base = SE3(UR3e_robotx, UR3e_roboty, UR3e_robotz) # Connect the UR3e to the base of the robot table
UR3e_robot.add_to_env(env)

# UR3e Table
RobotTable = part_mesh("RobotTable")
RobotTable_z = -0.62
RobotTable.T = UR3e_robot.base[0] * SE3(0, 0, RobotTable_z)
RobotTable_stopx = UR3e_robotx - 0.35
RobotTable_stopy = UR3e_roboty - 0.35
RobotTable_stopz = RobotTable_z + UR3e_robotz
RobotTable_stop = SE3(RobotTable_stopx, RobotTable_stopy, RobotTable_stopz)
env.add(RobotTable)

# env.hold()

## TRAJECTORIES

urtbl1 = UR3e_robot.q
urtbl2 = UR3e_robot.ikine_LM(SE3(final_grapeJuiceBottle_stop), UR3e_robot.q, 200, mask=[1, 1, 0, 0, 1, 0]).q
UR3e_stubot = jtraj(urtbl1, urtbl2, 200).q
for q in UR3e_stubot:
    UR3e_robot.q = q
    grapeJuiceBottle.T = UR3e_robot.fkine(UR3e_robot.q).A
    env.step(0.01)

grapeJuiceBottle.T = SE3(final_grapeJuiceBottle)

env.step()
env.hold()

