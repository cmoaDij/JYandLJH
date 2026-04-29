
# LeTac-MPC: Learning Model Predictive Control for Tactile-reactive Grasping

[arXiv](https://arxiv.org/abs/2403.04934) | [Summary Video](https://drive.google.com/file/d/1rDwg7dA3Wfhhb3rhry0cIfAxGli7WT7k/view?usp=drive_link)

Accepted by IEEE T-RO.

## LeTac-MPC training
```bash
$ python scripts/training.py
```
## Model-based controllers
For model-based controllers, see `model_based_pd.py` and `model_based_mpc.py`.

## Inference example
See `LeTac-MPC/README_infer.md` for a minimal inference script and usage.

## Dataset
See `LeTac-MPC/dataset`.

## BibTex

If you find this codebase useful, consider citing:

```bibtex
@ARTICLE{xuletac2024,
  author={Xu, Zhengtong and She, Yu},
  journal={IEEE Transactions on Robotics}, 
  title={{LeTac-MPC}: Learning Model Predictive Control for Tactile-Reactive Grasping}, 
  year={2024},
  volume={},
  number={},
  doi={10.1109/TRO.2024.3463470}}
```
