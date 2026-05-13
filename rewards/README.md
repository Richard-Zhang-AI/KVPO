# Reward Components

This directory contains the reward aggregation interface used by KVPO training.
Reward functions are configured through the `reward_components` section in the
training YAML files.

Each component returns per-video scores. KVPO normalizes branch rewards against
the anchor branch and combines the configured components by weight. The default
training configs use a mix of HPSv3, VideoAlign, motion smoothness, dynamic
degree, face consistency, and human-window quality rewards depending on the
target generator.

External model weights are not stored in this repository. Download the release
bundle from Hugging Face and place it under `checkpoints/` as described in the
root README. Optional dependencies such as `detectron2`, `insightface`, and
`hpsv3` are installed from `environment_kvpo.yml`; their upstream source trees
are intentionally not vendored here.

Example component block:

```yaml
reward_components:
  - name: video_hpsv3
    weight: 0.5
  - name: videoalign_mq
    weight: 1.0
  - name: motion_smoothness
    weight: 0.5
```
