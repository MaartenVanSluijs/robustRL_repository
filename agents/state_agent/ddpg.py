import torch
from torch.optim import Adam
import gymnasium as gym

from copy import deepcopy
import numpy as np
import wandb
from tqdm import trange

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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



def ddpg(env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=2000, epochs=1000, replay_size=int(1e6), gamma=0.99, 
         polyak=0.999, pi_lr=1e-4, q_lr=1e-4, batch_size=100, update_every=100, 
         noise_scale=0.1, SGLD_steps=5, target_eps=0.075, reg_factor=0.3,
         log_results=False):
    """
    Deep Deterministic Policy Gradient (DDPG)


    Args:
        env_name : Name of the Environment to be used.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        noise_scale (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name, render_mode="human")
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Set up Wandb run
    if log_results:
        config={
            "env": env_name,
            "agent_type": "state",
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "noise_scale": noise_scale,
            "batch_size": batch_size,
            "polyak": polyak,
            "seed": seed
            }
        run = wandb.init(
            project="robustRL_benchmark",
            group="train_state",
            config=config
            )

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
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

    # Gets the epsilon for regularization based on timestep
    def get_eps():
        total_steps = epochs*steps_per_epoch

        # In the first half we don't do any regularization
        if t < 0.5 * (total_steps):
            return 0
        # From 0.5 to 0.75 we linearly increase to the target eps
        elif t < 0.75 * (total_steps):
            index = t - (0.5*total_steps)
            frac = index / (0.25*total_steps)   
            return frac*target_eps              
        # After 0.75 we stay at target eps
        else:
            return target_eps
        
    def tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        x = np.asarray(x, dtype=np.float)
        x = torch.tensor(x, dtype=torch.float32)
        return x

    # Compute the SGLD policy regularizer
    def compute_regularizer(observation, action):
        robust_eps = get_eps()

        step_eps = torch.tensor(robust_eps / SGLD_steps, dtype=torch.float32) # TODO: We should actually still scale robust_eps to the state_range

        # upper and lower bounds for clipping
        phi = tensor(observation)
        adv_ub = phi + robust_eps
        adv_lb = phi - robust_eps
        
        # add uniform noise beween +/- scaled_robust_eps
        # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
        beta = 1e-5
        noise_factor = torch.sqrt(2 * step_eps) * beta
        noise = torch.randn_like(phi) * noise_factor

        # First SGLD step, the gradient is 0, so only need to add noise. Project to Linf box.
        adv_phi = (phi.clone() + noise.sign() * step_eps).detach().requires_grad_()
        
        for i in range(SGLD_steps):
            # Find a nearby state adv_phi that maximize the difference
            # We go through pi here instead of ac.act to get the gradient and to keep it a tensor
            adv_loss = (ac.pi(adv_phi) - action.detach()).pow(2).mean()
            # Need to clear gradients before the backward() for policy_loss
            adv_loss.backward()
            # Reduce noise at every step. We start at step 2.
            noise_factor = torch.sqrt(2 * step_eps) * beta / (i+2)
            # Project noisy gradient to step boundary.
            update = (adv_phi.grad + noise_factor * torch.randn_like(adv_phi)).sign() * step_eps
            adv_phi = adv_phi + update
            # clip into the upper and lower bounds, note that nothing changes when step_eps is still 0
            adv_phi = torch.max(adv_phi, adv_lb)
            adv_phi = torch.min(adv_phi, adv_ub).detach().requires_grad_()
            
        # We want to minimize the loss
        # Same here, we go through ac.pi instead of ac.act to get the gradient and tensor
        
        # I expected the new and old actions to be the same under an eps=0, this does not seem to be the case...
        action_reg_loss = (ac.pi(adv_phi) - action).pow(2).mean()
        regularizer =  reg_factor * action_reg_loss 
        return regularizer
    
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
        a = data['act']
        q_pi = ac.q(o, ac.pi(o))

        policy_regularizer = compute_regularizer(o, a)
        
        return -q_pi.mean() + policy_regularizer

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    def update(data):
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
        
        return loss_q.detach().numpy(), loss_pi.detach().numpy()

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    observation = o[0]
    
    # Main loop: collect experience in env and update/log each epoch
    for t in trange(total_steps):
        
        a = get_action(observation, noise_scale)

        # Step the env
        next_observation, r, terminated, truncated, info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Store experience to replay buffer
        replay_buffer.store(observation, a, r, next_observation, terminated)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        observation = next_observation

        # End of trajectory handling
        if terminated:
            if log_results:
                run.log({"Episode Return": ep_ret, "Episode Length": ep_len})
                run.log({"Episode Distance": info["x_position"]})
            o, ep_ret, ep_len = env.reset(), 0, 0
            observation = o[0]
            
        # Update handling
        if replay_buffer.size > batch_size and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                q_loss, pi_loss = update(data=batch)

                # Log to Wandb
                if log_results:
                    run.log({"q_loss": q_loss, "pi_loss": pi_loss})

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v5')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument("--log_results", type=bool, default=False)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    
    parser.add_argument("--polyak", type=float, default=0.999)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument('--target_eps', type=float, default=0.075)
    parser.add_argument('--reg_factor', type=float, default=0.3)
    args = parser.parse_args()

    ddpg(args.env_name, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[400,300]),
         gamma=args.gamma, seed=args.seed, epochs=args.num_epochs, log_results=args.log_results,
         polyak=args.polyak, noise_scale=args.noise_scale, batch_size=args.batch_size, replay_size=args.replay_size,
         target_eps=args.target_eps, reg_factor=args.reg_factor)
