# LeTac-MPC Inference Example

This is a minimal inference runner for the trained LeTac-MPC model.

## What it does
- Loads the checkpoint produced by `scripts/training.py`.
- Runs a forward pass on a single RGB image with a given gripper position and velocity.
- Prints the prediction sequence and the last step prediction.

## Quick start
```bash
python scripts/infer_example.py --checkpoint checkpoints/letac_mpc_checkpoint.pt --dummy
```

## Using a real image
```bash
python scripts/infer_example.py --checkpoint checkpoints/letac_mpc_checkpoint.pt --image /path/to/image.png \
  --gripper-p 30.0 --gripper-v 0.0
```

## Notes
- Image preprocessing matches `scripts/training.py`.
- If `--image` is not provided, use `--dummy`.
