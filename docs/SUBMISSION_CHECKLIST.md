# Copyright 2026 RTGTX7.

# Submission Checklist

Use this checklist before sharing the repository with a collaborator, attaching it to a paper, or packaging it as a research artifact.

## 1. Reproducibility

- [ ] exact training command recorded
- [ ] exact evaluation command recorded
- [ ] exact reward-search command recorded if search affected final weights
- [ ] dataset paths documented as placeholders, not hard-coded local machine paths
- [ ] checkpoint selection rule documented

## 2. Documentation

- [ ] top-level README points to valid docs
- [ ] one documentation index exists
- [ ] install, train, and evaluate paths are documented
- [ ] project-specific extensions are documented
- [ ] draft notes are clearly separated from primary docs

## 3. Code Organization

- [ ] canonical training script identified
- [ ] canonical evaluation script identified
- [ ] canonical reward wrapper identified
- [ ] experiment utilities documented
- [ ] duplicate or legacy-facing scripts are clearly labeled in docs

## 4. Paper Alignment

- [ ] method description matches the actual code path
- [ ] reward equations match the implemented reward wrapper
- [ ] encoder description matches the actual experiment setting
- [ ] baseline names in the paper match real training/evaluation variants

## 5. Ownership and Copyright

- [ ] original V-Max / Valeo copyright headers preserved
- [ ] new project-specific files include the correct local ownership notice

## 6. Recommended Minimal Artifact

At minimum, an external user should be able to:

1. install the environment
2. train one model
3. evaluate one checkpoint
4. understand where the custom reward and search logic live

If any of these steps is still unclear, the repository is not yet submission-ready.
