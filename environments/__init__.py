from gymnasium.envs.registration import register

register(
    id="RandomizedHopper-v0",
    entry_point="environments.random_hopper:RandomHopper",
)

register(
    id="RandomizedWalker-v0",
    entry_point="environments.random_hopper:RandomWalker",
)
