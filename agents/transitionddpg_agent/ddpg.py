from copy import deepcopy
from os import makedirs, getcwd, path
from argparse import ArgumentParser
from tqdm import trange

import numpy as np
import torch
from torch.optim import Adam
import wandb

from environments.robust_hopper import RobustHopper
from environments.robust_walker import RobustWalker2d
import core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, next_obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
class Trajectory:
    """
    Object for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.discounted_return = 0        

    def store(self, obs, next_obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.size += 1

def ddpg(env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
         psi_range=0.1, nr_trajectories=500, epsilon=10,
         num_epochs=1000, replay_size=int(1e6), gamma=0.99, max_trajectory_length=2000, num_updates=50,
         polyak=0.999, pi_lr=1e-4, q_lr=1e-4, batch_size=100, noise_scale=0.1,
         log_results=False, render=True):

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    if env_name == "Hopper-v5":
        env = RobustHopper()
    elif env_name == "Walker2d-v5":
        env = RobustWalker2d()

    env.render_mode = "human"

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Set up Wandb run
    if log_results:
        config={
            "env": env_name,
            "agent_type": "base",
            "epochs": num_epochs,
            "noise_scale": noise_scale,
            "batch_size": batch_size,
            "polyak": polyak,
            "seed": seed
            }
        run = wandb.init(
            project="robustRL_benchmark",
            group="train_base",
            config=config
            )

    # Method to return the directory in which to save the results of the training
    def get_directory():
        # Go to a models folder
        base_dir = getcwd() + '/agents/transitionddpg_agent/models/' + env_name + '/'

        base_dir += str(noise_scale) + "_" + str(batch_size) + "_" + str(polyak) + "/"

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
    
    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    def update():

        data = replay_buffer.sample_batch(batch_size)

        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        if log_results:
            run.log({"q_loss": loss_q, "pi_loss": loss_pi})

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def sample_trajectories() -> list[Trajectory]:
        trajectories = []

        for _ in range(nr_trajectories):

            trajectory = Trajectory(obs_dim=obs_dim, act_dim=act_dim, size=max_trajectory_length)
            
            t = 0
            episode_return = 0

            observation, _ = env.reset(options={
                "mass_scale": psi_range,
                "gravity_scale": psi_range,
                "friction_scale": psi_range})
            terminated = False
            truncated = False

            while not (terminated or truncated):

                action = get_action(observation, noise_scale)

                next_observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                trajectory.discounted_return += (gamma**t * reward)
                
                # save and log
                trajectory.store(observation, next_observation, action, reward, terminated)
                
                # Update obs (critical!)
                observation = next_observation
                t += 1

            if log_results:
                run.log({"Episode_length": t, "Episode_return": episode_return,
                        "Episode_distance": info["x_position"]})

            trajectories.append(trajectory)
            
        return trajectories

    # Main loop: generate trajectories, select the worst percentile, store and update
    for t in range(num_epochs):

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        
        trajectories = sample_trajectories()

        # Compute the epsilon-percentile return, default is 10th percentile
        percentile = np.percentile([trajectory.discounted_return for trajectory in trajectories], epsilon)

        # Filter trajectories
        trajectories = [
            traj for traj in trajectories if traj.discounted_return <= percentile
        ]
        
        # Add leftover trajectories to the replay buffer
        for trajectory in trajectories:
            for step in range(trajectory.size):
                replay_buffer.store(
                    trajectory.obs_buf[step],
                    trajectory.obs2_buf[step],
                    trajectory.act_buf[step],
                    trajectory.rew_buf[step],
                    trajectory.done_buf[step]
                )

        print(f"Buffer is filled with {replay_buffer.size}/{replay_buffer.max_size} ({round(replay_buffer.size/replay_buffer.max_size*100,2)}%)")

        for _ in range(num_updates):
            update()

        # Save the current model
        torch.save(ac.pi.state_dict(), base_dir + "/pi.pt")
        torch.save(ac.q.state_dict(), base_dir + "/q.pt")

    # After the total steps save the model again
    print("Finished training the agent, writing the model to file...")
    torch.save(ac.pi.state_dict(), base_dir + "/pi.pt")
    torch.save(ac.q.state_dict(), base_dir + "/q.pt")
    print("Succesfully written the model to file")

    env.close()

            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v5')
    parser.add_argument('--seed', '-s', type=int, default=0)
    
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_epochs', type=int, default=10000)
    
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--polyak", type=float, default=0.99)

    parser.add_argument('--nr_trajectories', type=int, default=500)
    parser.add_argument("--epsilon", type=int, default=10)
    parser.add_argument("--psi_range", type=float, default=0.2)

    parser.add_argument("--log_results", type=bool, default=False)
    parser.add_argument("--render", type=bool, default=True)


    args = parser.parse_args()

    ddpg(args.env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[400,300]), 
         gamma=args.gamma, seed=args.seed, num_epochs=args.num_epochs, log_results=args.log_results,
         polyak=args.polyak, noise_scale=args.noise_scale, batch_size=args.batch_size,
         nr_trajectories=args.nr_trajectories, epsilon=args.epsilon, psi_range=args.psi_range,
         render=args.render
         )
