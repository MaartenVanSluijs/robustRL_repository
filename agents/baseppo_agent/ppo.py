import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import wandb
from tqdm import trange

import core

from os import getcwd, makedirs, path


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env_name, seed, num_epochs, render, log_results,
        gamma, 
        clip_ratio, gae_lambda, target_kl,
        train_pi_iters=80, train_v_iters=80):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    PI_LR = 3e-4
    V_LR = 1e-3
    EPOCH_LENGTH = 4000

    # Set up Wandb run
    if log_results:
        config={
            "env": env_name,
            "agent_type": "baseppo",
            "epochs": num_epochs,
            "clip_ratio": clip_ratio,
            "target_kl": target_kl,
            "gae_lambda": gae_lambda,
            "seed": seed
            }
        run = wandb.init(
            project="robustRL_benchmark",
            group="train_baseppo",
            config=config
            )
        
    # Method to return the directory in which to save the results of the training
    def get_directory():
        # Go to a models folder
        base_dir = getcwd() + '/agents/baseppo_agent/models/' + env_name + '/'

        base_dir += str(target_kl) + "_" + str(clip_ratio) + "_" + str(gae_lambda) + "/"

        # Check which run number this is
        run_number = 0
        while path.exists(base_dir + str(run_number)):
            run_number += 1
        base_dir += str(run_number)

        # Return complete directory
        return base_dir
    
    base_dir = get_directory()
    print(f"Writing model to: {base_dir}")
    makedirs(base_dir)

    # Instantiate environment
    env = gym.make(env_name, render_mode="human" if render else "rgb_array")
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = core.MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=[400,300])

    # Set up experience buffer
    buffer = PPOBuffer(obs_dim, act_dim, EPOCH_LENGTH, gamma, gae_lambda)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=PI_LR)
    vf_optimizer = Adam(ac.v.parameters(), lr=V_LR)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    
    def update():
        data = buffer.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        if log_results:
            run.log({"Stopping iteration": i})
        
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        kullback_leibler, entropy, clip_fraction = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        if log_results:
            run.log({"pi_loss": pi_l_old, "v_loss": v_l_old,
                    "kullback_leibler": kullback_leibler, "entropy": entropy,
                    "clip_fraction": clip_fraction})
                    

    # Main loop: collect experience in env and update/log each epoch
    for epoch in trange(num_epochs):
        # Prepare for interaction with environment
        observation, _ = env.reset()
        episode_return, episode_length = 0, 0

        for t in range(EPOCH_LENGTH):
            
            action, v, logp = ac.step(torch.as_tensor(observation, dtype=torch.float32))

            next_observation, reward, terminated, _, info = env.step(action)
            episode_return += reward
            episode_length += 1

            # save and log
            buffer.store(observation, action, reward, v, logp)
            
            # Update obs (critical!)
            observation = next_observation

            if terminated:
                
                if log_results:
                    run.log({"Episode_Length": episode_length,
                             "Episode_Distance": info["x_position"],
                             "Episode_Return": episode_return})
                
                buffer.finish_path(0)
                observation, _ = env.reset()
                episode_return, episode_length = 0, 0

        if not terminated:
            _, v, _ = ac.step(torch.as_tensor(observation, dtype=torch.float32))
            buffer.finish_path(v)

        # Perform PPO update!
        update()

        torch.save(ac.pi.state_dict(), base_dir + "/pi.pt")
        torch.save(ac.v.state_dict(), base_dir + "/v.pt")

    # Save the models
    print("Finished training the agent, writing the model to file...")
    torch.save(ac.pi.state_dict(), base_dir + "/pi.pt")
    torch.save(ac.v.state_dict(), base_dir + "/v.pt")
    print("Succesfully written the model to file")

        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v5')
    parser.add_argument('--seed', '-s', type=int, default=41)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument("--log_results", type=bool, default=False)
    parser.add_argument("--render", type=bool, default=True)
    
    parser.add_argument("--target_kl", type=float, default=0.03) #0.03, 0.003
    parser.add_argument("--clip_ratio", type=float, default=0.3) #0.1, 0.2, 0.3
    parser.add_argument("--gae_lambda", type=float, default=0.9) #0.9, 0.95, 1.0

    parser.add_argument('--gamma', type=float, default=0.99)
    
    args = parser.parse_args()

    ppo(args.env_name, seed=args.seed, num_epochs=args.num_epochs, render=True, log_results=args.log_results,
        gamma=args.gamma, 
        clip_ratio=args.clip_ratio, gae_lambda=args.gae_lambda, target_kl=args.target_kl)