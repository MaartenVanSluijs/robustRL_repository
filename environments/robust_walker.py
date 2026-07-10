from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
import mujoco
import numpy as np

# Made assisted with ChatGPT
class RobustWalker2d(Walker2dEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Cache IDs
        self.torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.left_foot_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_left"
        )
        self.left_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_left"
        )
        self.right_foot_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_right"
        )
        self.right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot_right"
        )

        # Cache defaults
        self.default_mass = self.model.body_mass[self.torso_id]
        self.default_gravity = self.model.opt.gravity.copy()
        self.default_friction_left = self.model.geom_friction[self.left_foot_geom_id].copy()
        self.default_friction_right = self.model.geom_friction[self.right_foot_geom_id].copy()

        # Scales (modifiable before reset)
        self.mass_scale = 0.0
        self.gravity_scale = 0.0
        self.friction_scale = 0.0
        self.action_noise = 0.0
        self.observation_noise = 0.0

    def step(self, action):
        # Robust action
        action += (np.random.normal(0, self.action_noise) * action)
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])

        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()

        # Robust observation
        observation += (np.random.normal(0, self.observation_noise) * observation)
        observation = np.clip(observation, self.observation_space.low[0], self.observation_space.high[0])

        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset(self, *, seed=None, options=None):
        # Allow passing scales via Gymnasium reset options
        if options is not None:
            self.mass_scale = options.get("mass_scale", self.mass_scale)
            self.gravity_scale = options.get("gravity_scale", self.gravity_scale)
            self.friction_scale = options.get("friction_scale", self.friction_scale)
            self.action_noise = options.get("action_noise", self.action_noise)
            self.observation_noise = options.get("observation_noise", self.observation_noise)

        return super().reset(seed=seed)

    def reset_model(self):
        # --- Mass ---
        self.model.body_mass[self.torso_id] = (
            self.default_mass
            * np.random.uniform(1 - self.mass_scale, 1 + self.mass_scale)
        )

        # --- Gravity ---
        self.model.opt.gravity[:] = self.default_gravity
        self.model.opt.gravity[2] *= np.random.uniform(
            1 - self.gravity_scale, 1 + self.gravity_scale
        )

        # --- Friction (both feet) ---
        friction = self.default_friction_left[0] * np.random.uniform(
            1 - self.friction_scale, 1 + self.friction_scale
        )

        self.model.geom_friction[self.left_foot_geom_id, :] = [
            friction, friction * 0.1, friction * 0.01
        ]
        self.model.geom_friction[self.right_foot_geom_id, :] = [
            friction, friction * 0.1, friction * 0.01
        ]

        mujoco.mj_forward(self.model, self.data)
        return super().reset_model()
