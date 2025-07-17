import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

coxa_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "coxa")
pata_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pata")
cube_qpos_addr = model.joint(name="cube_free_joint").qposadr

L1 = 55
Y_MAX = 50
Y_MIN = 15
X_MAX = 20
X_MIN = -20
x_pos = X_MIN
y_pos = Y_MIN
time_start = time.time()
demo_positions = [[X_MIN, Y_MIN], [X_MAX, Y_MIN], [X_MAX, Y_MAX], [X_MIN, Y_MAX]]
counter_demo = 0

def move_cube_to(x, y, z):
    data.qpos[cube_qpos_addr + 0] = x
    data.qpos[cube_qpos_addr + 1] = y
    data.qpos[cube_qpos_addr + 2] = z

def ik_2d(_x, _y):
    global L1
    c = np.sqrt(_x**2 + _y**2)
    alpha_1 = np.arccos(c / (2 * L1)) + np.arctan2(_x,_y)
    alpha_2 = np.arccos(-c**2 / (2 * L1**2) + 1)
    return alpha_1, alpha_2

with mujoco.viewer.launch_passive(model, data) as viewer:
    #start = time.time()
    while True:
        # x_pos += 0.1
        # y_pos += 0.1
        if counter_demo >= len(demo_positions):
            counter_demo = 0
        if time.time() - time_start > 1:
            time_start = time.time()
            x_pos, y_pos = demo_positions[counter_demo]
            counter_demo += 1
        x,y = x_pos, y_pos
        print(f"x: {x}, y: {y}")
        if x_pos > X_MAX:
            x_pos = X_MIN
        if y_pos > Y_MAX:
            y_pos = Y_MIN
        move_cube_to((-x_pos)/1000,(-y_pos-L1)/1000+0.01,0.01)
        alpha_1, alpha_2 = ik_2d(x_pos,2*L1 - y_pos)
        data.ctrl[coxa_id] = alpha_1
        data.ctrl[pata_id] = alpha_2
        time.sleep(0.01)
        mujoco.mj_step(model, data)
        viewer.sync()
