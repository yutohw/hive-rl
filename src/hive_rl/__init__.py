"""Self-driving race car trained with PPO inside SideFX Houdini.

Modules:
    config      paths, environment constants, PPO hyperparameters, seeding
    comms       file-based messaging between trainer and workers
    curriculum  track selection schedule
    model       ActorCritic network and TanhNormal distribution
    ppo         schedules, GAE, transition cap, PPO update
    worker      hython entry point: plays episodes in a Houdini scene
    trainer     hython entry point: gathers rollouts and updates the policy
"""
