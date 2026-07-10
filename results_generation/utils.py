import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import trange

from environments.robust_hopper import RobustHopper
from environments.robust_walker import RobustWalker2d

class Model(nn.Module):
    def __init__(self, state_dict, agent_type, act_dim):
        super().__init__()

        self.agent_type = agent_type

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

        state_dict = {k.split(".", 1)[-1]: v
        for k, v in state_dict.items()}

        self.dist_agents = ["transition_agent", "baseppo_agent", "stateppo_agent", "actionppo_agent"]

        self.pi = self._build_and_load(state_dict)

    def _build_and_load(self, state_dict):
        layers = []
        weight_keys = [k for k in state_dict if k.endswith("weight")]

        for i, k in enumerate(weight_keys):
            out_dim, in_dim = state_dict[k].shape
            layers.append(nn.Linear(in_dim, out_dim))
            if self.agent_type in self.dist_agents:
                layers.append(nn.Tanh() if i < len(weight_keys) - 1 else nn.Identity())
            else:
                layers.append(nn.ReLU() if i <len(weight_keys) - 1 else nn.Tanh())

        pi = nn.Sequential(*layers)
        pi_state_dict = {k: v for k, v in state_dict.items() if k in pi.state_dict()}
        pi.load_state_dict(pi_state_dict)
        return pi

    def _distribution(self, obs):
        mu = self.pi(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs):
        if self.agent_type in self.dist_agents:
            return self._distribution(obs).sample().detach().numpy()
        return self.pi(obs).detach().numpy()
    
class Evaluation():

    def __init__(self, env_name, render=False, num_episodes=100):
        
        # Set atttributes
        self.render = render
        self.num_episodes = num_episodes
        self.env_name = env_name

        self.max_episode_length = 2000

        self.noise_types = []

        # Deduce environment
        if self.env_name == "Hopper-v5":
            self.env = RobustHopper()
            self.env.render_mode = "human" if self.render else "rgb_array"
        
        elif self.env_name == "Walker2d-v5":
            self.env = RobustWalker2d()
            self.env.render_mode = "human" if self.render else "rgb_array"
    
    def set_noise(self, noise_types: list[str], noise_end, noise_step):
        self.noise_range = np.linspace(0, noise_end, noise_step)
        self.noise_types = noise_types

    def determine_noise_type(self, agent_type):
        if "base" in agent_type:
            return "base"
        elif "state" in agent_type:
            return "state"
        elif "transition" in agent_type:
            return "transition"
        elif "action" in agent_type:
            return "action"
        elif "disturbance" in agent_type:
            return "disturbance"
        else:
            return None

    def evaluate_agent(self, agent: Model):
        noise_results = {}

        for noise_level in self.noise_range:

            print(f"Using noise: {noise_level}")

            options = {
                "mass_scale": noise_level if "transition" in self.noise_types else 0.0,
                "gravity_scale": noise_level if "transition" in self.noise_types else 0.0,
                "friction_scale": noise_level if "transition" in self.noise_types else 0.0,
                "action_noise": noise_level if "action" in self.noise_types else 0.0,
                "observation_noise": noise_level if "state" in self.noise_types else 0.0
            }

            print(options)
            
            noise_results[noise_level] = {}

            noise_results[noise_level]["cumulative_reward"] = []
            noise_results[noise_level]["distance"] = []

            for episode in trange(self.num_episodes):

                state, info = self.env.reset(seed=41, options=options)
                cumulative_reward = 0
                observation = torch.from_numpy(state).float()
                terminated, truncated = False, False

                for step in range(self.max_episode_length):

                    action = agent.forward(torch.as_tensor(observation, dtype=torch.float32))
                    
                    next_observation, reward, terminated, truncated, info = self.env.step(action)

                    observation = torch.from_numpy(next_observation).float()
                    cumulative_reward += reward

                    if terminated or truncated:
                        break
                
                noise_results[noise_level]["cumulative_reward"].append(cumulative_reward)
                noise_results[noise_level]["distance"].append(info["x_position"])

        return noise_results