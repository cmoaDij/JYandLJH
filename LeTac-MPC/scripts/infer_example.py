import argparse
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms

from functions import ResCNNEncoder, MPClayer


def build_transform(res_size=224):
    return transforms.Compose(
        [
            transforms.Resize([res_size, res_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2]),
        ]
    )


def load_checkpoint(checkpoint_path, device, cnn_encoder, mpc_layer):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cnn_encoder.load_state_dict(checkpoint["cnn_encoder_state"])
    mpc_layer.load_state_dict(checkpoint["mpc_layer_state"])


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def main():
    parser = argparse.ArgumentParser(description="LeTac-MPC inference example")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/letac_mpc_checkpoint.pt",
        help="Path to checkpoint file.",
    )
    parser.add_argument("--image", type=str, help="Path to input RGB image.")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use a dummy random image if --image is not provided.",
    )
    parser.add_argument("--gripper-p", type=float, default=30.0, help="Gripper position.")
    parser.add_argument("--gripper-v", type=float, default=0.0, help="Gripper velocity.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cnn_encoder = ResCNNEncoder(hidden1=128, hidden2=128, dropP=0.15, outputDim=20).to(device)
    mpc_layer = MPClayer(nHidden=20, eps=1e-4, nStep=15, del_t=1 / 25).to(device)

    load_checkpoint(args.checkpoint, device, cnn_encoder, mpc_layer)

    cnn_encoder.eval()
    mpc_layer.eval()

    if args.image:
        image = load_image(args.image)
    elif args.dummy:
        image = Image.fromarray((255 * torch.rand(224, 224, 3)).byte().numpy())
    else:
        raise ValueError("Provide --image or use --dummy")

    transform = build_transform()
    X = transform(image).unsqueeze(0).to(device)
    gripper_p = torch.tensor([args.gripper_p], device=device)
    gripper_v = torch.tensor([args.gripper_v], device=device)

    with torch.no_grad():
        emb = cnn_encoder(X)
        pred = mpc_layer(emb, gripper_p, gripper_v)

    print("Prediction shape:", tuple(pred.shape))
    print("Prediction sequence:", pred[0].tolist())


if __name__ == "__main__":
    main()
