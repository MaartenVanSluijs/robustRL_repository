from argparse import ArgumentParser
from copy import deepcopy
import os
from tqdm import trange

import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import wandb

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

    def store(self, obs, act, rew, next_obs, done):
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

def ddpg(env_name, seed, log_results, render, num_epochs,
         noise_scale, batch_size, polyak,
         alpha, ratio,
         replay_size, gamma):
    
    EPOCH_LENTGH = 2000
    HIDDEN_SIZES = [400, 300]
    UPDATE_FREQUENCY = 100
    PI_LR = 1e-4
    Q_LR = 1e-4

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name, render_mode="human" if render else "rgb_array")
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Set up Wandb run
    if log_results:
        config={
            "env": env_name,
            "agent_type": "state",
            "epochs": num_epochs,
            "steps_per_epoch": EPOCH_LENTGH,
            "noise_scale": noise_scale,
            "batch_size": batch_size,
            "polyak": polyak,
            "seed": seed
            }
        run = wandb.init(
            project="robustRL_benchmark",
            group="train_action",
            config=config
            )

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = core.MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=HIDDEN_SIZES)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Method to return the directory in which to save the results of the training
    def get_directory():
        # Go to a models folder
        base_dir = os.getcwd() + '/agents/action_agent/models/' + env_name + '/'
        
        # Include the method and alpha settings
        base_dir += str(noise_scale) + '_' + str(batch_size) + '_' + str(polyak) + '/'

        # Check which run number this is
        run_number = 0
        while os.path.exists(base_dir + str(run_number)):
            run_number += 1
        base_dir += str(run_number)

        # Return complete directory
        return base_dir
    
    base_dir = get_directory()
    print(f"Writing model to: {base_dir}")
    os.makedirs(base_dir)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=PI_LR)
    adversary_optimizer = Adam(ac.adversary.parameters(), lr=PI_LR)
    q_optimizer = Adam(ac.q.parameters(), lr=Q_LR)

    num_steps = num_epochs * EPOCH_LENTGH
    observation, episode_length, episode_return = env.reset()[0], 0, 0

    def get_action(observation):
        with torch.no_grad():
            agent_action = ac.pi(torch.as_tensor(observation, dtype=torch.float32)).numpy()
            adversary_action = ac.adversary(torch.as_tensor(observation, dtype=torch.float32)).numpy()
        
            action = ((1-alpha) * agent_action) + (alpha*adversary_action)
            noisy_action = action + (noise_scale * np.random.randn(act_dim))
        return np.clip(noisy_action, -act_limit, act_limit)
    
    def critic_update(data):
        q_optimizer.zero_grad()
        # Unpack data
        observation, action, reward, next_observation, terminated = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # Compute state-action values
        q = ac.q(observation, action)

        # Compute next actions
        next_actions = (1-alpha)*ac_targ.pi(next_observation) + alpha*ac_targ.adversary(next_observation)

        # Bellman backup for Q function
        with torch.no_grad():
            # Compute expected state-action values
            q_pi_targ = ac_targ.q(next_observation, next_actions)
            # Compute Bellman operator
            backup = reward + gamma * (1 - terminated) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()
        loss_q.backward()
        q_optimizer.step()

        return loss_q.detach().numpy()

    def actor_update(data):
        pi_optimizer.zero_grad()
        observation = data["obs"]

        with torch.no_grad():
            adversary_action = ac_targ.adversary(observation)
        
        action = (1-alpha)*ac.pi(observation) + alpha*adversary_action

        loss_pi = -ac.q(observation, action).mean()
        loss_pi.backward()
        pi_optimizer.step()

        return loss_pi.detach().numpy()

    def adversary_update(data):
        adversary_optimizer.zero_grad()
        observation = data["obs"]

        with torch.no_grad():
            actor_action = ac_targ.pi(observation)

        action = (1-alpha)*actor_action + alpha*ac.adversary(observation)

        loss_adversary = ac.q(observation, action).mean()
        loss_adversary.backward()
        adversary_optimizer.step()

        return loss_adversary.detach().numpy()

    def update_parameters(update_adversary):
        batch = replay_buffer.sample_batch(batch_size=batch_size)

        # Update Q once
        critic_loss = critic_update(batch)

        # Freeze Q network for computational optimization
        for p in ac.q.parameters():
            p.requires_grad = False

        # Update Actor once
        if not update_adversary:
            actor_loss = actor_update(batch)
            adversary_loss = 0

        # Update Adversary if update_adversary is true
        else:
            adversary_loss = adversary_update(batch)
            actor_loss = 0

        # Unfreeze Q network
        for p in ac.q.parameters():
            p.requires_grad = True

        # Polyak average
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        
        return critic_loss, actor_loss, adversary_loss

    for step in trange(num_steps):

        action = get_action(observation)
        
        # Step the env
        next_observation, reward, terminated, truncated, info = env.step(action)
        episode_length += 1
        episode_return += reward

        # Store experience to replay buffer
        replay_buffer.store(observation, action, reward, next_observation, terminated)

        # Super critical, easy to overlook step: make sure to update most recent observation!
        observation = next_observation

        # End of trajectory handling
        if terminated:
            if log_results:
                run.log({"Episode Return": episode_return, "Episode Length": episode_length})
                run.log({"Episode Distance": info["x_position"]})
            observation, episode_return, episode_length = env.reset()[0], 0, 0

        if replay_buffer.size > batch_size and step % UPDATE_FREQUENCY == 0:

            for update_step in range(UPDATE_FREQUENCY):
                
                update_adversary = True if update_step % ratio == 0 else False

                actor_loss, adversary_loss, critic_loss = update_parameters(update_adversary)

                # Log to Wandb
                if log_results:
                    run.log({"q_loss": critic_loss, "pi_loss": actor_loss, "adversary_loss": adversary_loss})
                            
        # End of epoch handling
        if (step+1) % EPOCH_LENTGH == 0:
            torch.save(ac.pi.state_dict(), base_dir + "/pi.pt")
            torch.save(ac.q.state_dict(), base_dir + "/q.pt")
            torch.save(ac.adversary.state_dict(), base_dir + "/adversary.pt")

    # After the total steps save the model again
    print("Finished training the agent, writing the model to file...")
    torch.save(ac.pi.state_dict(), base_dir + "/pi.pt")
    torch.save(ac.q.state_dict(), base_dir + "/q.pt")
    torch.save(ac.adversary.state_dict(), base_dir + "/adversary.pt")
    print("Succesfully written the model to file")

    env.close()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--env_name", type=str, default="Hopper-v5")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--log_results", type=bool, default=False)
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=1000)

    parser.add_argument("--noise_scale", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--polyak", type=float, default=0.99)

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ratio", type=int, default=10)

    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    ddpg(env_name=args.env_name, seed=args.seed, log_results=args.log_results, render=args.render, num_epochs=args.num_epochs,
         noise_scale=args.noise_scale, batch_size=args.batch_size, polyak=args.polyak,
         alpha=args.alpha, ratio=args.ratio,
         replay_size=args.replay_size, gamma=args.gamma)