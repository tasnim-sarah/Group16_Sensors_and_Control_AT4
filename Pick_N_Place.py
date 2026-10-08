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

# Control Panel
controlPanel = part_mesh("ControlPanel")
controlPanel.T = SE3(3, 0.75, 1.5)
controlPanel.scale = (0.2, 0.2, 0.2)
#controlPanel.T = trotz(pi/2)
env.add(controlPanel)

# Emergency Stop
eStop = part_mesh("emergencyStopButton")
eStop.scale = (0.35, 0.35, 0.35)
eStop.T = SE3(2.35, 1.7, 0.6)
env.add(eStop)

# Fences
fence1 = part_mesh("fenceAssemblyGreenRectangle4x8x2.5m")
fence1.scale = (0.5, 1, 1)
fence1.T = SE3(2.35, 3.5, -1)
env.add(fence1)

# Baby
baby = part_mesh("baby")
baby.scale = (0.65, 0.65, 0.65)
baby.T = transl(2.78, 2, 0.47)
env.add(baby)

# Chair
chair = part_mesh("tableRound0.3x0.3x0.3m")
chair.scale = (1.25, 1.25, 1.63)
chair.T = transl(2.95, 2, 0)
env.add(chair)

# Safety Helmet
SafetyHelmet = part_mesh("SafetyHelmet")
SafetyHelmet.scale = (0.63, 0.63, 0.6)
SafetyHelmet.T = transl(2.82, 2.21, 0.73)
env.add(SafetyHelmet)


# Plate
plate = part_mesh("Plate")
plate.T = SE3(2.22, 2, 0.6)
plate.scale = (0.7, 0.7, 0.7)
env.add(plate)

# Grape Juice Glass 
juiceGlass = part_mesh("WineCup")
juiceGlass.T = SE3(2.05, 2.28, 0.6)
env.add(juiceGlass)

# Grape Juice Bottle
grapeJuiceBottle = part_mesh("WineBottle")
grapeJuiceBottle_x = 3.885
grapeJuiceBottle_y = 1.458
grapeJuiceBottle_z = 0.025
grapeJuiceBottle.T = SE3(grapeJuiceBottle_x, grapeJuiceBottle_y, grapeJuiceBottle_z)
grapeJuiceBottle_stopx = grapeJuiceBottle_x - 0.4
grapeJuiceBottle_stopy = grapeJuiceBottle_y - 0.4
grapeJuiceBottle_stop = SE3(grapeJuiceBottle_stopx, grapeJuiceBottle_stopy, 0)
grapeJuiceBottle_targetz = grapeJuiceBottle_z + 0.4
grapeJuiceBottle_target = SE3(grapeJuiceBottle_x, grapeJuiceBottle_y, grapeJuiceBottle_targetz)
grapeJuiceBottle_keyx = grapeJuiceBottle_x - 1.5
grapeJuiceBottle_keyy = grapeJuiceBottle_y - 1.5
grapeJuiceBottle_keyz = grapeJuiceBottle_z + 1
grapeJuiceBottle_key = SE3(grapeJuiceBottle_keyy, grapeJuiceBottle_keyy, grapeJuiceBottle_keyz)
env.add(grapeJuiceBottle)

# Final Position for Grape Juice Bottle
final_grapeJuiceBottle_stop = SE3(1.95, 6, 0.9)
final_grapeJuiceBottle = SE3(2.05, 6, 0.6)

# Bottle Crate
bottleCrate = part_mesh("BottleCrate") 
bottleCrate.T = SE3(1.5, 4, 0)
bottleCrate.scale = (1.25, 1.25, 1.25)
env.add(bottleCrate)

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

# TurtleBot
turtleBot = Turtlebot3Waffle()
turtleBot.base = SE3(1.5, 3, 0)
turtleBot.add_to_env(env)

# Student Robot
# DH Links for Student Robot
l1 = DHLink(d=0.1450, a=0, alpha=pi/2, qlim=[-pi, pi])
l2 = DHLink(d=0, a=-0.2500, alpha=0, qlim=[-pi, pi])
l3 = DHLink(d=0, a=-0.1800, alpha=0, qlim=[-pi, pi])
l4 = DHLink(d=0.1050, a=0, alpha=pi/2, qlim=[-pi, pi])
l5 = DHLink(d=0.0860, a=0, alpha=-pi/2, qlim=[-pi, pi])
l6 = DHLink(d=0.0860, a=0, alpha=0, qlim=[-pi, pi])
studentRobot = DHRobot([l1, l2, l3, l4, l5, l6], name='my_robot')
studentRobot.q = [-pi/2, -pi/2, 0, -pi/2, pi/2, 0]  # Define initial joint state for robot
           
# Creating simple cylindrical geometry for each link of the DHRobot (but this time all links will be blue)
cyl_viz = CylindricalDHRobotPlot(studentRobot, cylinder_radius=0.05, color="#9dd1c6")
studentRobot = cyl_viz.create_cylinders()
studentRobot.base = turtleBot.fkine(np.zeros(1)).A  # Connect the student robot to the base of the turtle robot
env.add(studentRobot)

# env.hold()

## TRAJECTORIES

turtleBot_bottle = rtb.ctraj(SE3(turtleBot.base), SE3(grapeJuiceBottle_stop), 200)
for t in turtleBot_bottle:
    turtleBot.base = t
    studentRobot.base = turtleBot.fkine(np.zeros(1)).A 
    env.step(0.01)

# env.hold()

gp_new1 = studentRobot.q
gp_new2 = studentRobot.ikine_LM(grapeJuiceBottle_target, q0=np.zeros([1,6]), mask=[1, 1, 1, 0, 1, 0]).q
to_bottle = jtraj(gp_new1, gp_new2, 200).q
for q in to_bottle:
    studentRobot.q = q
    env.step(0.01)

# env.hold()

grapeJuiceBottle.T = studentRobot.fkine(studentRobot.q).A

gp_new1 = studentRobot.q
gp_new2 = studentRobot.ikine_LM(grapeJuiceBottle_key, q0=np.zeros([1,6]), mask=[1, 1, 1, 0, 1, 0]).q
to_bottle = jtraj(gp_new1, gp_new2, 200).q
for q in to_bottle:
    studentRobot.q = q
    grapeJuiceBottle.T = studentRobot.fkine(studentRobot.q).A
    env.step(0.01)
grapeJuiceBottle.T = studentRobot.fkine(studentRobot.q).A

turtleBot_UR3e = rtb.ctraj(SE3(turtleBot.base), SE3(RobotTable_stop), 200)
for t in turtleBot_UR3e:
    turtleBot.base = t
    studentRobot.base = turtleBot.fkine(np.zeros(1)).A 
    grapeJuiceBottle.T = studentRobot.fkine(studentRobot.q).A
    env.step(0.01)

grapeJuiceBottle_pose = SE3(grapeJuiceBottle.T) * SE3(0, 0, 0.35)
urstb1 = UR3e_robot.q
urstb2 = UR3e_robot.ikine_LM(SE3(grapeJuiceBottle_pose), UR3e_robot.q, 200, mask=[1, 1, 0, 0, 1, 0]).q
UR3e_stubot = jtraj(urstb1, urstb2, 200).q
for q in UR3e_stubot:
    UR3e_robot.q = q
    grapeJuiceBottle.T = studentRobot.fkine(studentRobot.q).A
    env.step(0.01)
grapeJuiceBottle.T = UR3e_robot.fkine(UR3e_robot.q).A

urtbl1 = UR3e_robot.q
urtbl2 = UR3e_robot.ikine_LM(SE3(final_grapeJuiceBottle_stop), UR3e_robot.q, 200, mask=[1, 1, 0, 0, 1, 0]).q
UR3e_stubot = jtraj(urtbl1, urtbl2, 200).q
for q in UR3e_stubot:
    UR3e_robot.q = q
    grapeJuiceBottle.T = UR3e_robot.fkine(UR3e_robot.q).A
    env.step(0.01)

grapeJuiceBottle.T = SE3(final_grapeJuiceBottle)

# jtraj(turtleBot.base[1], grapeJuiceBottle.T[1], steps).q

env.step()
env.hold()

