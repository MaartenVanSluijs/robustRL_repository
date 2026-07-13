### Robust reinforcement learning
This repository is made to evaluate the robustness of reinforcement learning (RL).

#### Description
The repo provides a centralized system of evaluating RL agents on MuJoCo environments, with functionality for adding mismatches in the evaluation environments. Furthermore, it provides a set of pre-implemented robust RL agents built upon the PPO and DDPG algorithms. Lastly, this repository also includes trained models and evaluation results from several experiments evaluating the robustness of the agents.

---
#### Installation
To install the repository follow the following steps:

```sh
git clone https://www.github.com/your-repo/RobustRL_Repository.git
cd RobustRL_Repository
```

Create a virtual environment:

```sh
python -m venv env
./env/Scripts/activate
```

and install the requirements:

```sh
pip install -r requirements.txt
```

#### CLI usage
```sh
# Train an agent (e.g. base agent)
python agents/base_agent/ddpg.py --env_name Hopper-v5 ---num_epochs 1000

# Evaluate agents with a single type of mismatch present on the Hopper task
python results_generation/single_noise.py agents/base_agent/models/Hopper-v5/0.1_100_0.99
```