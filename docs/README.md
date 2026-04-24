# Copyright 2026 RTGTX7.

# V-Max Documentation Index

This directory collects the user-facing documentation needed to run, extend, and evaluate this repository in its current form.

The repository remains based on **V-Max** and keeps the original V-Max / Valeo code ownership and copyright notices in the source tree. The files listed below document the current workflow and the project-specific extensions added in this workspace.

## Start Here

- [Usage Guide](./USAGE_GUIDE.md): practical setup, training, evaluation, and experiment workflow
- [Command Reference](./COMMAND_REFERENCE.md): canonical commands and project-specific experiment commands
- [Repository Organization Guide](./REPOSITORY_ORGANIZATION.md): canonical vs auxiliary code paths
- [Configuration Guide](./1_config.md): Hydra configuration structure and override patterns
- [Training Pipeline](./2_training_pipeline.md): training entry points, checkpoints, resume modes, and search scripts
- [Observation Guide](./3_observation.md): structured observation design and extractor behavior
- [Reward Guide](./4_reward.md): reward wrappers, multiplicative reward behavior, and safe-bubble integration
- [Metrics Guide](./5_metrics.md): evaluation metrics and aggregate reporting
- [Evaluation Guide](./6_evaluation.md): how to run benchmark-style evaluation and where outputs are written
- [Multi-Machine Search](./7_multi_machine_search.md): distributed reward-search workflow
- [Extension Inventory](./8_new_functions_inventory.md): implementation inventory for newer additions
- [Submission Checklist](./SUBMISSION_CHECKLIST.md): practical pre-release checklist for academic artifact sharing

## Recommended Reading Order

For a new user, the shortest path is:

1. Read [Usage Guide](./USAGE_GUIDE.md)
2. Read [Configuration Guide](./1_config.md)
3. Read [Training Pipeline](./2_training_pipeline.md)
4. Read [Evaluation Guide](./6_evaluation.md)

If you are working on research changes, then continue with:

5. [Observation Guide](./3_observation.md)
6. [Reward Guide](./4_reward.md)
7. [Metrics Guide](./5_metrics.md)

## Research Notes and Draft Material

The following files are useful for internal experimentation, paper writing, or figure generation, but they are not the primary user entry points:

- [experimental_design_draft.md](./experimental_design_draft.md)
- [ppo_reward_math_notes.md](./ppo_reward_math_notes.md)
- [network_structure_tikz.tex](./network_structure_tikz.tex)
- [network_structure_sample.dot](./network_structure_sample.dot)
- `docs/notebooks/`

## Scope of This Documentation

These documents aim to answer four practical questions:

- How do I install and run the repository?
- How do I train a model?
- How do I evaluate a checkpoint?
- Where are the project-specific extensions, especially reward shaping and experiment search?

They do not replace the source code. For implementation details, follow the file references in each guide.
