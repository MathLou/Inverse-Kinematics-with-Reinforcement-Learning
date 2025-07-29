from stable_baselines3 import PPO, SAC
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import viewer
x_target, y_target = 0, 0

class PlanarIKEnv(gym.Env):
    def __init__(self, xml_path="scene.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        #robot
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site")
        self.coxa_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "coxa")
        self.pata_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pata")
        # cube
        self.cube_qpos_addr = self.model.joint(name="cube_free_joint").qposadr
        # Link length
        self.L = 55

        # Limits
        self.X_MIN, self.X_MAX = -30/1000, 30/1000
        self.Y_MIN, self.Y_MAX = -50/1000, 50/1000

        # Action space: angle deltas or absolute position commands
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        # Observation: [theta1, theta2, x_ee, y_ee, x_target, y_target]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.target = np.array([0.0, 0.0])
        
    def set_target(self, x, y):
        self.target = np.array([x, y])  
    def move_cube_to(self, x, y, z):
        self.data.qpos[self.cube_qpos_addr + 0] = x
        self.data.qpos[self.cube_qpos_addr + 1] = y
        self.data.qpos[self.cube_qpos_addr + 2] = z

    def reset(self, seed=42, options=None):
        global x_target, y_target
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # # Gera um novo alvo aleatório SEMPRE
        # self.target = np.array([
        #     np.random.uniform(self.X_MIN, self.X_MAX),
        #     np.random.uniform(self.Y_MIN, self.Y_MAX)
        # ])
        self.target = np.array([x_target/1000, y_target/1000])  # Define a posição do alvo como (0, 0)

        # Move o cubo para a nova posição
        self.move_cube_to(self.target[0], self.target[1], 0.01)

        # Posição inicial aleatória do robô
        self.data.ctrl[self.coxa_id] = np.random.uniform(-5, 5) * np.pi / 180
        self.data.ctrl[self.pata_id] = np.random.uniform(-5, 5) * np.pi / 180

        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {}


    def _get_obs(self):
        pos = self.data.site_xpos[self.site_id]
        theta1 = self.data.ctrl[self.coxa_id]
        theta2 = self.data.ctrl[self.pata_id]
        return np.array([theta1, theta2, pos[0], pos[1], self.target[0], self.target[1]], dtype=np.float32)

    def step(self, action):
        scalar = 2*np.pi
        self.data.ctrl[self.coxa_id] = action[0] * scalar
        self.data.ctrl[self.pata_id] = action[1] * scalar

        mujoco.mj_step(self.model, self.data)

        ee_pos_x, ee_pos_y , _ = self.data.site_xpos[self.site_id]
        target_x, target_y = self.target
        distance = ((ee_pos_x - target_x)**2 + (ee_pos_y - target_y)**2)**0.5
        if distance*1000 < 10:  # Se a distância for muito pequena, recompensa alta
            reward = np.exp(-np.abs(distance))  # Penaliza a distância
        else:
            reward = -distance
        terminated = distance*1000 < 10
        truncated = terminated  # ou True se quiser limitar o número de steps

        obs = self._get_obs()
        info = {"distance": distance}

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if not hasattr(self, "viewer"):
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if hasattr(self, "viewer"):
            self.viewer.close()
#i = 0
env = PlanarIKEnv()
model = SAC.load("sac_IK_planar_v5", env=env, device="cuda",verbose=1)
while True:
    amplitude_x = 50
    for i in range(-amplitude_x,amplitude_x,5):
        x_target, y_target = i, 50  # Define the target position
        start_time = time.time()
        obs, _ = env.reset()
        done = False
        while time.time() - start_time < 1:  # Executa por 60 segundos
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"target [mm]: {x_target,y_target}, position [mm]: {1000*obs[2:4]}, reward: {reward:.2f}")
            env.render()
            time.sleep(0.01)  # adiciona um pequeno delay se necessário para visualização
            #done = terminated or truncated
