# Legacy scripts (not part of the modernised PyTorch stack)

These files depend on external multi-agent RL frameworks (`pymarl`, `smac`) that
are not maintained for current Python and are out of scope for this modernisation:

- `multiagent_seir.py` (original `epcontrol/multiagent/seir.py`): SMAC `MultiAgentEnv` wrapper.
- `seir_pymarl.py`: PyMARL launcher.
- `seir_environment_multi_run_ppo_policy.py`: applies one policy per district; kept for reference.

The single-district and joint multi-district PPO paths in `scripts/` cover the
same experiments using `stable-baselines3` and do not require these dependencies.
