from stable_baselines3 import PPO
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import viewer

class PlanarIKEnv(gym.Env):
    def __init__(self, xml_path="scene.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site")
        self.coxa_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "coxa")
        self.pata_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pata")

        self.L = 55  # Link length

        # Limits
        self.X_MIN, self.X_MAX = -20, 20
        self.Y_MIN, self.Y_MAX = 15, 50

        # Action space: angle deltas or absolute position commands
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        # Observation: [theta1, theta2, x_ee, y_ee, x_target, y_target]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.target = np.array([0.0, 0.0])
        
    def set_target(self, x, y):
        self.target = np.array([x, y])  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Somente escolha um novo alvo se ainda não foi definido
        if not hasattr(self, "target") or self.target is None:
            self.target = np.array([
                np.random.uniform(self.X_MIN, self.X_MAX),
                np.random.uniform(self.Y_MIN, self.Y_MAX)
            ])

        self.data.ctrl[self.coxa_id] = 0.0
        self.data.ctrl[self.pata_id] = 0.0

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.data.site_xpos[self.site_id]
        theta1 = self.data.ctrl[self.coxa_id]
        theta2 = self.data.ctrl[self.pata_id]
        return np.array([theta1, theta2, pos[0], pos[1], self.target[0], self.target[1]], dtype=np.float32)

    def step(self, action):
        scalar = np.pi
        self.data.ctrl[self.coxa_id] = action[0] * scalar
        self.data.ctrl[self.pata_id] = action[1] * scalar

        mujoco.mj_step(self.model, self.data)

        ee_pos_x, ee_pos_y , _ = self.data.site_xpos[self.site_id]
        target_x, target_y = self.target
        distance = ((ee_pos_x - target_x)**2 + (ee_pos_y - target_y)**2)**0.5

        reward = np.exp(-np.abs(distance))

        terminated = distance < 0.01
        truncated = False  # ou True se quiser limitar o número de steps

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

env = PlanarIKEnv()

model = PPO.load("ppo_IK_planar", env=env, device="cpu")

targets = [[20,-50]]
for x, y in targets:
    env.set_target(x, y)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"target [mm]: {env.target}, position [mm]: {1000*obs[2:4]}")
        env.render()
        time.sleep(0.01)  # adiciona um pequeno delay se necessário para visualização

