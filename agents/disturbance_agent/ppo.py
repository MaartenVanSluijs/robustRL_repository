from argparse import ArgumentParser
from os import getcwd, path, makedirs

import ppo_core as core
from environments.robust_hopper import RobustHopper
from environments.robust_walker import RobustWalker2d

import torch
from torch.optim import Adam
import numpy as np
import wandb
from tqdm import trange

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, gae_lambda=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.gae_lambda = gamma, gae_lambda
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
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
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


def ppo(env_name, seed, log_results, render, num_epochs,
        target_kl, clip_ratio, gae_lambda,
        adversary_strength, actor_updates, adversary_updates, warmup_updates,
        gamma, train_pi_iters=80, train_v_iters=80):
    
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    HIDDEN_SIZES=[400,300]
    PI_LR = 3e-4
    VF_LR = 1e-3
    EPOCH_LENGTH = 4000
    FORCE=1

    # Set up Wandb run
    if log_results:
        config={
            "env": env_name,
            "agent_type": "disturbance_agent",
            "epochs": num_epochs,
            "clip_ratio": clip_ratio,
            "target_kl": target_kl,
            "gae_lambda": gae_lambda
            }
        run = wandb.init(
            project="robustRL_benchmark",
            group="train_disturbance",
            config=config
            )

    # Instantiate environment
    if env_name == "Hopper-v5":
        env = RobustHopper()
        adversary_space = np.array([0,0])
    elif env_name == "Walker2d-v5":
        env = RobustWalker2d()
        adversary_space = np.array([0,0,0,0])

    env.render_mode = "human" if render else "rgb_array"

    obs_dim = env.observation_space.shape

    # Create actor-critic module
    actor = core.MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=HIDDEN_SIZES)
    adversary = core.MLPActorCritic(env.observation_space, adversary_space, hidden_sizes=HIDDEN_SIZES)

    # Method to return the directory in which to save the results of the training
    def get_directory():
        # Go to a models folder
        base_dir = getcwd() + '/agents/disturbance_agent/models/' + env_name + '/'

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

    # Set up optimizers for policy and value function
    actor_optimizer = Adam(actor.pi.parameters(), lr=PI_LR)
    actor_v_optimizer = Adam(actor.v.parameters(), lr=VF_LR)
    
    adversary_optimizer = Adam(adversary.pi.parameters(), lr=PI_LR)
    adversary_v_optimizer = Adam(adversary.v.parameters(), lr=VF_LR)

    def apply_force(nu):
        nu *= (FORCE*adversary_strength)
        if env_name == "Hopper-v5":
            env.data.xfrc_applied[env.foot_id] = [nu[0], 0, nu[1], 0, 0, 0]
        elif env_name == "Walker2d-v5":
            env.data.xfrc_applied[env.left_foot_id] = [nu[0], 0, nu[1], 0, 0, 0]
            env.data.xfrc_applied[env.right_foot_id] = [nu[2], 0, nu[3], 0, 0, 0]
    
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, agent: core.MLPActorCritic):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = agent.pi(obs, act)
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
    def compute_loss_v(data, agent: core.MLPActorCritic):
        obs, ret = data['obs'], data['ret']
        return ((agent.v(obs) - ret)**2).mean()

    def update(buffer: PPOBuffer, agent: core.MLPActorCritic, pi_optimizer, vf_optimizer):

        data = buffer.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, agent)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, agent).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, agent)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        # if log_results:
        #     run.log({"Stopping iteration": i})

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, agent)
            loss_v.backward()
            vf_optimizer.step()

        # Log changes from update
        # kullback_leibler, entropy, clip_fraction = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # if log_results:
        #     run.log({"pi_loss": pi_l_old, "v_loss": v_l_old,
        #             "kullback_leibler": kullback_leibler, "entropy": entropy,
        #             "clip_fraction": clip_fraction})

    def fill_buffer(buffer: PPOBuffer, agent_role: str):
        observation, _ = env.reset()
        episode_return, episode_length, terminated = 0, 0, False

        # First collect samples with adversary fixed
        for step in range(EPOCH_LENGTH):

            pi, v_pi, logp_pi = actor.step(torch.as_tensor(observation, dtype=torch.float32))
            nu, v_nu, logp_nu = adversary.step(torch.as_tensor(observation, dtype=torch.float32))

            apply_force(nu)

            next_observation, reward, terminated, _, info = env.step(pi)
            if agent_role == "actor":
                buffer.store(observation, pi, reward, v_pi, logp_pi)
            elif agent_role == "adversary":
                buffer.store(observation, nu, -reward, v_nu, logp_nu)

            observation = next_observation
            
            episode_return += reward
            episode_length += 1

            if terminated:
                buffer.finish_path(0)

                if log_results:
                    run.log({"Episode_length": episode_length, 
                             "Episode_return": episode_return,
                             "Episode_distance": info["x_position"]})

                observation, _ = env.reset()
                episode_return, episode_length, terminated = 0, 0, False

            if agent_role == "actor":
                _, v_pi, _ = actor.step(torch.as_tensor(observation, dtype=torch.float32))
                buffer.finish_path(v_pi)
            elif agent_role == "adversary":
                _, v_nu, _ = actor.step(torch.as_tensor(observation, dtype=torch.float32))
                buffer.finish_path(v_nu)
        return buffer
    
    print("Warming up the agent...")
    for warmup_round in trange(warmup_updates):
        empty_buffer = PPOBuffer(obs_dim, env.action_space.shape, EPOCH_LENGTH, gamma=gamma, gae_lambda=gae_lambda)
        buffer = fill_buffer(empty_buffer, agent_role="actor")

        update(buffer, agent=actor, pi_optimizer=actor_optimizer, vf_optimizer=actor_v_optimizer)
    print("Finished agent warming up, introducing adversary...")

    for epoch in trange(num_epochs):
        for actor_step in range(actor_updates):
            empty_buffer = PPOBuffer(obs_dim, env.action_space.shape, EPOCH_LENGTH, gamma=gamma, gae_lambda=gae_lambda) 
            buffer = fill_buffer(empty_buffer, agent_role="actor")

            update(buffer, agent=actor, pi_optimizer=actor_optimizer, vf_optimizer=actor_v_optimizer)

        for adversary_step in range(adversary_updates):
            empty_buffer = PPOBuffer(obs_dim, adversary_space.shape, EPOCH_LENGTH, gamma=gamma, gae_lambda=gae_lambda) 
            buffer = fill_buffer(empty_buffer, agent_role="adversary")

            update(buffer, agent=adversary, pi_optimizer=adversary_optimizer, vf_optimizer=adversary_v_optimizer)

        torch.save(actor.pi.state_dict(), base_dir + "/pi.pt")
        torch.save(actor.v.state_dict(), base_dir + "/v_pi.pt")
        
        torch.save(adversary.pi.state_dict(), base_dir + "/nu.pt")
        torch.save(adversary.v.state_dict(), base_dir + "/v_nu.pt")

    # Save the models
    print("Finished training the agent, writing the model to file...")
    torch.save(actor.pi.state_dict(), base_dir + "/pi.pt")
    torch.save(actor.v.state_dict(), base_dir + "/v_pi.pt")

    torch.save(adversary.pi.state_dict(), base_dir + "/nu.pt")
    torch.save(adversary.v.state_dict(), base_dir + "/v_nu.pt")
    print("Succesfully written the model to file")

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--env_name", type=str, default="Hopper-v5")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--log_results", type=bool, default=False)
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=100) # I think we do 300 now to get the same as other models

    parser.add_argument("--target_kl", type=float, default=0.03) #0.03, 0.003
    parser.add_argument("--clip_ratio", type=float, default=0.3) #0.1, 0.2, 0.3
    parser.add_argument("--gae_lambda", type=float, default=0.9) #0.9, 0.95, 1.0

    parser.add_argument("--adversary_strength", type=int, default=1) # We have no idea what a reasonable value is
    parser.add_argument("--actor_updates", type=int, default=10)
    parser.add_argument("--adversary_updates", type=int, default=5)
    parser.add_argument("--warmup_updates", type=int, default=100)

    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    ppo(env_name=args.env_name, seed=args.seed, log_results=args.log_results, render=args.render, num_epochs=args.num_epochs,
        target_kl=args.target_kl, clip_ratio=args.clip_ratio, gae_lambda=args.gae_lambda,
        adversary_strength=args.adversary_strength, actor_updates=args.actor_updates, adversary_updates=args.adversary_updates, warmup_updates=args.warmup_updates,
        gamma=args.gamma)