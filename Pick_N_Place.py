import time
from math import pi
import numpy as np
import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from ir_support import CylindricalDHRobotPlot
from ir_support.robots import UR3e
from ir_support_extra_parts.parts import part_mesh
from ir_support_extra_robots.robots import Turtlebot3Waffle


# STUDENTBOT6 DH PARAMETERS  - SID: 25829007

d1 = 0.1450 #   d1  = shoulder height off the base (offset along joint-1's z axis)
a2 = -0.3000 #   a2  = "upper arm" length (link length of joint 2)
a3 = -0.1900 #   a3  = "forearm" length (link length of joint 3) NEGATIVE LOCAL AXIS ORIENTATION AFTER ALPHA TWIST
d4 = 0.1250
d5 = 0.0700
d6 = 0.0635

#   d4,d5,d6 = wrist offsets


# CREATE STUDENTBOT6

#   joint 1: alpha=+90  -> twists you off the vertical base axis
#   joints 2,3: alpha=0 -> PARALLEL axes -> planar "shoulder/elbow" pair,
#                          this is what actually gives you reach
#   joints 4,5,6: alpha alternates +90/-90/0 -> builds the wrist so the
#                 tool can re-orient independent of position

links = [
    rtb.RevoluteDH(d=d1, a=0, alpha=pi / 2, qlim=[-2 * pi, 2 * pi]),  # base can rotate all around without collidng with itslef 
    rtb.RevoluteDH(d=0, a=a2, alpha=0, qlim=[-pi / 2, pi / 2]), # shoulder and elbow pair - parallell - restricte dto 90 so it dioenst fold into itself 
    rtb.RevoluteDH(d=0, a=a3, alpha=0, qlim=[-2 * pi / 3, 2 * pi / 3]), # elbow part - parrlllel to shoulder - 120 degrees safe spot 
    rtb.RevoluteDH(d=d4, a=0, alpha=pi / 2, qlim=[-2 * pi, 2 * pi]), # below are all wrists they can rotate anyway without worry of folding or hittign itslef 
    rtb.RevoluteDH(d=d5, a=0, alpha=-pi / 2, qlim=[-2 * pi, 2 * pi]),
    rtb.RevoluteDH(d=d6, a=0, alpha=0, qlim=[-2 * pi, 2 * pi]),
]

studentbot = rtb.DHRobot(links, name="StudentBot6")

print("StudentBot6 joint limits:")
print(studentbot.qlim)
print("StudentBot6 joints at limits:", studentbot.islimit(studentbot.q))


# Swift environment 

env = swift.Swift()
env.launch(realtime=True)


#Studentbot6 - visuals
cyl_viz = CylindricalDHRobotPlot(studentbot, cylinder_radius=0.05, color="purple")
studentbot.scale = [0.3, 0.3, 0.3]
studentbot = cyl_viz.create_cylinders()
studentbot.q = np.zeros(6)
env.add(studentbot)


# Turtlebot 

turtlebot = Turtlebot3Waffle()
turtlebot.add_to_env(env)

#Toolbox pixkup 

PICKUP_X = 1.0
PICKUP_Y = 1.2
TOOLBOX_Z = 0.13

PARK_OFFSET_X = 0.25   # 1.0 - 0.75
PARK_OFFSET_Y = 0.35   # 1.1 - 0.75

PARK_X = PICKUP_X - PARK_OFFSET_X
PARK_Y = PICKUP_Y - PARK_OFFSET_Y

HOVER_HEIGHT_PICKUP = 0.50 - TOOLBOX_Z
DESCEND_HEIGHT_PICKUP = 0.30 - TOOLBOX_Z

def toolbox_pose():
    return SE3(PICKUP_X, PICKUP_Y, TOOLBOX_Z)

def pickup_hover_pose():
    return SE3(PICKUP_X, PICKUP_Y, TOOLBOX_Z + HOVER_HEIGHT_PICKUP)

def pickup_descend_pose():
    return SE3(PICKUP_X, PICKUP_Y, TOOLBOX_Z + DESCEND_HEIGHT_PICKUP)


SAFETY_OFFSET_X = 0.0
SAFETY_OFFSET_Y = 0.1   # 1.2 - 1.1

safety_person = part_mesh("SafetyPerson")
safety_person.scale = [0.3, 0.3, 0.3]
safety_person.T = (SE3(PICKUP_X + SAFETY_OFFSET_X, PICKUP_Y + SAFETY_OFFSET_Y, 0) * SE3.Rz(pi)).A
env.add(safety_person)

# environemnt and safety

toolbox = part_mesh("Toolbox")
toolbox.scale = [0.3, 0.3, 0.3]
toolbox.T = toolbox_pose().A   
env.add(toolbox)


workbench = part_mesh("Workbench")
workbench.scale = [0.3, 0.3, 0.3]
workbench.T = (SE3(1.93, 1.9, 0) * SE3.Rz(pi / 2)).A
env.add(workbench)

emergencyB = part_mesh("emergencyStopbutton")
emergencyB.scale = [0.3, 0.3, 0.3]
emergencyB.T = (SE3(1.93, 1.9, 0.275)) # just incase robot crashes out
env.add(emergencyB)

# create UR3e robot 
ur3 = UR3e()
ur3.base = SE3(1.93, 2.2, 0.275)  # on table 
ur3.q = np.zeros(6)
ur3.add_to_env(env)



target_crate = part_mesh("MilkCrate")
target_crate.scale = [0.6, 0.6, 0.6]
target_crate.T = SE3(1.60, 2.10, 0.0).A  # front of tabel 
env.add(target_crate)


collection_worker = part_mesh("personMaleConstruction")
collection_worker.scale = [0.3, 0.3, 0.3]
collection_worker.T = (SE3(2.0, 1.0, 0) * SE3.Rz(pi / 2)).A # he is walkign to collect the toolbox 
env.add(collection_worker)

light_tower = part_mesh("LightTower")
light_tower.scale = [0.3, 0.3, 0.3]
light_tower.T = SE3(2.25, 1.3, 0).A # worker waits for light to go green 
env.add(light_tower)

print("STATUS: Robot cell active - worker must wait")


# The cones - i liek them 

def add_cone(x, y):
    cone = part_mesh("TrafficCone")
    cone.scale = [0.3, 0.3, 0.3]
    cone.T = SE3(x, y, 0).A
    env.add(cone)


cone_spacing = 0.5
scene_size = 2.5

for x in np.arange(0, scene_size + 0.01, cone_spacing):
    add_cone(x, 0)
    add_cone(x, scene_size)

for y in np.arange(cone_spacing, scene_size, cone_spacing):
    add_cone(0, y)
    add_cone(scene_size, y)



# STUDENTBOT6 ON TURTLEBOT

studentbot.base = turtlebot.base * SE3(0, 0, 0.15) # - a bit up

# 2D collision avoidance for the mobile base. for each obstacle it checks distance from the (x,y) point to the obstacle centre - purely geometric 
def avoid_obstacles_2d(x, y, obstacles):
    """obstacles: list of (centre_x, centre_y, keep_out_radius)."""
    for ox, oy, radius in obstacles:
        dx, dy = x - ox, y - oy
        dist = np.hypot(dx, dy)
        if dist < radius and dist > 1e-6:
            scale = radius / dist
            x = ox + dx * scale
            y = oy + dy * scale
    return x, y


# Keep-out circles built from the actual placed scene objects, so they stay correct if you move a prop's position later. distance is radius + clearence 
OBSTACLES = [
    (workbench.T[0, 3], workbench.T[1, 3], 0.65),          
    (collection_worker.T[0, 3], collection_worker.T[1, 3], 0.35),  
    (safety_person.T[0, 3], safety_person.T[1, 3], 0.35),  
]



# DRIVE TURTLEBOT TO PICKUP ZONE and avoid hititgnnayhting 

DRIVE_STEPS_1 = 100

for i in range(DRIVE_STEPS_1):
    fraction = i / (DRIVE_STEPS_1 - 1)
    x, y = avoid_obstacles_2d(PARK_X * fraction, PARK_Y * fraction, OBSTACLES)
    turtlebot.base = SE3(x, y, 0)
    studentbot.base = turtlebot.base * SE3(0, 0, 0.15)

    env.step(0.05)

# Note: This motion is NOT IK and NOT jtraj.
# The TurtleBot base is just a free SE3 pose, so we move it by linearly interpolating its x,y position 
# The avoid_obstacles_2d() function only nudges the path if the base enters a keep‑out circle.




# STUDENTBOT6 PICKUP - IK


# ikine_LM = numerical Levenberg-Marquardt solver: start from seed q0, measure the pose error against the target, use the Jacobian to
#  step q towards the target, repeat until it converges.

pickup_hover = pickup_hover_pose()
pickup_pose = pickup_descend_pose()

#solving for 
# StudentBot6 joint angles to hover above the toolbox
sol_hover = studentbot.ikine_LM(
    pickup_hover,
    q0=studentbot.q, # IK starts guessing 
    mask=[1, 1, 1, 0, 0, 0] # only position matters 
)

if not sol_hover.success:
    raise RuntimeError(f"StudentBot hover IK failed: {sol_hover.reason}")

# solvin for StudentBot6 joint angles to descend onto the toolbox
sol_pickup = studentbot.ikine_LM(
    pickup_pose,
    q0=sol_hover.q, # keeps soldved poses close - based off previuos 
    mask=[1, 1, 1, 0, 0, 0]
)

if not sol_pickup.success:
    raise RuntimeError(f"StudentBot pickup IK failed: {sol_pickup.reason}")

# jtraj fits a quintic polynomial per joint between the start and end q, giving zero velocity/acceleration at both endpoints - used when we dont need a specifc path 
# moving from join to join configuration using smooth jointrejectory 
for q in rtb.jtraj(studentbot.q, sol_hover.q, 60).q:
    studentbot.q = q
    env.step(0.05)

for q in rtb.jtraj(studentbot.q, sol_pickup.q, 40).q:
    studentbot.q = q
    env.step(0.05)

# FORWARD KINEMATICS used here. fkine(q) computes the end-effector pose from the CURRENT joint angles

# Capture the toolbox's pose relative to the gripper at the exact instant of grasp, so it can be attached( no gripper)
toolbox_in_student_tool = studentbot.fkine(studentbot.q).inv() * SE3(toolbox.T)
# toolbox in wold coordinate inverse to coordniate relatibve to gripper then times by toolbox pose

# Raise the held toolbox back to the hover pose using smooth trajectory 
for q in rtb.jtraj(studentbot.q, sol_hover.q, 30).q:
    studentbot.q = q
    # FK every frame: recompute tool pose, re-attach toolbox to it.
    toolbox.T = (studentbot.fkine(studentbot.q) * toolbox_in_student_tool).A
    env.step(0.05)



#  STUDENTBOT6 TRANSPORTS TOOLBOX TO UR3e STATION
def solve_ur3_world(target_world, q0, name):
    """Solve UR3e IK using a world-frame target (its base is internal)."""
    solution = ur3.ikine_LM(
        target_world, q0=q0, mask=[1, 1, 1, 0, 0, 0],
        ilimit=1000, slimit=100
    )
    if not solution.success:
        raise RuntimeError(f"UR3e {name} IK failed: {solution.reason}")
    return solution

# Keep the mobile robot clear of the workbench while moving the handover
# point just inside the UR3e workspace.

station_x = 1.53
station_y = 1.46
DRIVE_STEPS_2 = 90

# Predict the exact final handover pose, including the same floor-obstacle adjustment used in the drive loop

final_x, final_y = avoid_obstacles_2d(station_x, station_y, OBSTACLES)
drive_delta = SE3(
    final_x - turtlebot.base.t[0], final_y - turtlebot.base.t[1], 0
)
planned_toolbox_world = drive_delta * SE3(toolbox.T)
planned_handover_hover = SE3(
    planned_toolbox_world.t[0], planned_toolbox_world.t[1],
    planned_toolbox_world.t[2] + 0.20
)
q_home = ur3.q.copy()
sol_handover_hover = solve_ur3_world(
    planned_handover_hover, q_home, "handover-hover"
)
ur3_transport_traj = rtb.jtraj(
    q_home, sol_handover_hover.q, DRIVE_STEPS_2
).q

for i in range(DRIVE_STEPS_2):
    fraction = i / (DRIVE_STEPS_2 - 1)

    x = PARK_X + (station_x - PARK_X) * fraction
    y = PARK_Y + (station_y - PARK_Y) * fraction
    x, y = avoid_obstacles_2d(x, y, OBSTACLES)

    turtlebot.base = SE3(x, y, 0)
    studentbot.base = turtlebot.base * SE3(0, 0, 0.15)
    studentbot.q = sol_hover.q   # arm stays up 
    ur3.q = ur3_transport_traj[i]

    # FK again: toolbox stays glued to StudentBot6's gripper while it travels.
    toolbox.T = (studentbot.fkine(studentbot.q) * toolbox_in_student_tool).A

    env.step(0.05)

print("Arrived at UR3e station")



# UR3e WORLD-FRAME IK


def solve_ur3_world(target_world, q0, name):
    
    solution = ur3.ikine_LM(
        target_world,
        q0=q0,
        mask=[1, 1, 1, 0, 0, 0],
        ilimit=1000,
        slimit=100
    )

    if not solution.success:
        raise RuntimeError(f"UR3e {name} IK failed: {solution.reason}")

    return solution



# UR3e HANDOVER


# The initial joint state is a known, collision-free home pose. 
q_home = ur3.q.copy()

toolbox_position = toolbox.T[:3, 3]

handover_hover_world = SE3(
    toolbox_position[0],
    toolbox_position[1],
    toolbox_position[2] + 0.20
)

handover_in_ur3_base = ur3.base.inv() * handover_hover_world
print(
    "UR3e handover target in base frame:", handover_in_ur3_base.t,
    "distance:", np.linalg.norm(handover_in_ur3_base.t),
    "reach:", ur3.reach
)

handover_grasp_world = SE3(
    toolbox_position[0],
    toolbox_position[1],
    toolbox_position[2] + 0.15
)

sol_handover_hover = solve_ur3_world(
    handover_hover_world,
    q_home,
    "handover-hover"
)

for q in rtb.jtraj(ur3.q, sol_handover_hover.q, 30).q:
    ur3.q = q
    env.step(0.05)

sol_handover = solve_ur3_world(
    handover_grasp_world,
    sol_handover_hover.q,
    "handover"
)

for q in rtb.jtraj(ur3.q, sol_handover.q, 20).q:
    ur3.q = q
    env.step(0.05)

# Toolbox pose relative to UR3e end effector while held
# Keep toolbox directly below UR3e tool for a stable visual grasp.
TOOLBOX_TOOL_OFFSET = ur3.fkine(ur3.q).t[2] - toolbox.T[2, 3]

def update_toolbox_from_ur3():
    # NOTE: no `end=` kwarg here either - see solve_ur3_world() note above.
    # FK again: this is the UR3e side of the same "attach by FK" pattern
    # used for StudentBot6 in Section 9/10.
    tool_pose = ur3.fkine(ur3.q)
    toolbox.T = SE3(
        tool_pose.t[0], tool_pose.t[1],
        tool_pose.t[2] - TOOLBOX_TOOL_OFFSET
    ).A

print("UR3e picked up toolbox")



# UR3e PLACEMENT

crate_position = target_crate.T[:3, 3]

# This is the final TOOLBOX pose inside the crate.
TOOLBOX_DROP_HEIGHT = 0.10
HOVER_HEIGHT = 0.25

toolbox_drop_world = SE3(
    crate_position[0],
    crate_position[1],
    crate_position[2] + TOOLBOX_DROP_HEIGHT
)

# Calculate the UR3e tool pose that preserves the captured grasp transform.
drop_tool_world = SE3(
    toolbox_drop_world.t[0],
    toolbox_drop_world.t[1],
    toolbox_drop_world.t[2] + TOOLBOX_TOOL_OFFSET
)
hover_drop_world = SE3(
    drop_tool_world.t[0],
    drop_tool_world.t[1],
    drop_tool_world.t[2] + HOVER_HEIGHT
)

# Leave the handover vertically first, then cross above the workbench.  

travel_height = max(0.75, hover_drop_world.t[2] + 0.20)
handover_clear_world = SE3(
    handover_grasp_world.t[0], handover_grasp_world.t[1], travel_height
)
sol_handover_clear = solve_ur3_world(
    handover_clear_world, sol_handover.q, "handover-clear"
)

for q in rtb.jtraj(ur3.q, sol_handover_clear.q, 25).q:
    ur3.q = q
    update_toolbox_from_ur3()
    env.step(0.05)

crate_clear_world = SE3(
    hover_drop_world.t[0], hover_drop_world.t[1], travel_height
)
sol_crate_clear = solve_ur3_world(
    crate_clear_world, sol_handover_clear.q, "crate-clear"
)

for q in rtb.jtraj(ur3.q, sol_crate_clear.q, 30).q:
    ur3.q = q
    update_toolbox_from_ur3()
    env.step(0.05)

sol_hover_drop = solve_ur3_world(
    hover_drop_world,
    sol_crate_clear.q,
    "crate-hover"
)

for q in rtb.jtraj(ur3.q, sol_hover_drop.q, 30).q:
    ur3.q = q
    update_toolbox_from_ur3()
    env.step(0.05)

sol_drop = solve_ur3_world(
    drop_tool_world,
    sol_hover_drop.q,
    "crate-drop"
)

for q in rtb.jtraj(ur3.q, sol_drop.q, 20).q:
    ur3.q = q
    update_toolbox_from_ur3()
    env.step(0.05)

# force the toolbox to the intended crate position.
toolbox.T = toolbox_drop_world.A
env.step(0.5)

# Move UR3e back above the crate after release.
for q in rtb.jtraj(ur3.q, sol_hover_drop.q, 60).q:
    ur3.q = q
    env.step(0.05)
 
 env.hold()

print("STATUS: Task complete - worker may collect crate")
print("Task complete: toolbox placed in MilkCrate")

