import argparse
import os
import json

from utils import Evaluation
from utils import Model

import torch


def get_file_paths(folder):
    file_paths = [] 
    folder_path = os.getcwd() + "/" + folder
    for run_number in os.listdir(folder_path):

        model_path = os.getcwd() + "/" + folder
        model_path += "/" + run_number
        model_path += "/" + "pi.pt"

        file_paths.append(model_path)
    return file_paths

def get_agents(folder, agent_type, env_name):
    # This is necessary for the gaussian distribution std
    if env_name == "Hopper-v5":
        action_dim = 3
    else:
        action_dim = 6
    
    file_paths = get_file_paths(folder)
    agents={}
    for path in file_paths:
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    
        agent = Model(state_dict, agent_type, action_dim)
        agent.load_state_dict(state_dict, strict=False)
        agents[path.split("/")[-2]] = agent
    return agents


def evaluate_agents(folder: str, render: bool, num_episodes: int):

    # Deduce the agent type
    agent_type = folder.split("/")[1] # Takes the form "action_agent"

    # Deduce the environment
    if "Hopper-v5" in folder:
        env_name = "Hopper-v5"
    elif "Walker2d-v5" in folder:
        env_name = "Walker2d-v5"

    # Get the agents
    agents = get_agents(folder, agent_type, env_name)

    # Set the noise types
    noise_types = ["transition", "action", "state"]

    # Initialize the evaluation suite.
    evaluation = Evaluation(env_name, render, num_episodes)

    results = {}
    for noise_type in noise_types:

        evaluation.set_noise([noise_type], 1.0, 10)

        per_noise_results = {}

        # Run each agent on the environment with the specified noise
        for run_number, agent in agents.items():
            print(f"Evaluating agent nr: {run_number}")
            noise_results = evaluation.evaluate_agent(agent)

            per_noise_results[run_number] = noise_results

        results[noise_type] = per_noise_results

    save_name = f"singlenoise_{agent_type}_{env_name}"

    with open(save_name+'.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("folder", type=str)
    argparser.add_argument("--render", type=bool, default=False)
    argparser.add_argument("--num_episodes", type=int, default=100)
    
    args = argparser.parse_args()

    evaluate_agents(folder=args.folder, render=args.render, num_episodes=args.num_episodes)