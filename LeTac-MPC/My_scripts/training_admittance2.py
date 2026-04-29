import os
import re
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import random
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.functions import *

parser = argparse.ArgumentParser(description="Train LeTac-MPC with admittance2 dataset")
group = parser.add_mutually_exclusive_group()
group.add_argument("--new-run", action="store_true", help="Force a new training run (ignore checkpoints)")
group.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
args = parser.parse_args()

CNN_hidden1, CNN_hidden2 = 128, 128
CNN_embed_dim = 20
res_size = 224
dropout_p = 0.15

epochs = 2000
batch_size = 16
learning_rate = 1e-4
eps = 1e-4
nStep = 15
del_t = 1 / 25

data_path = "dataset_admittance2"

transform = transforms.Compose([
    transforms.Resize([res_size, res_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])
])


def parse_trial_target_mm(trial_name):
    match = re.search(r"_gp_([-+]?\d*\.?\d+)mm", trial_name)
    if match is None:
        return None
    return float(match.group(1))


def parse_image_gp_mm(image_name):
    match = re.search(r"_gp_([-+]?\d*\.?\d+)mm", image_name)
    if match is not None:
        return float(match.group(1))

    match = re.search(r"_([-+]?\d*\.?\d+)mm\.(jpg|jpeg|png)$", image_name, re.IGNORECASE)
    if match is not None:
        return float(match.group(1))

    return None


def read_admittance2_data(root_path):
    selected_all_names = []
    output_p = []
    grip_posi_num = []
    grip_vel_num = []

    if not os.path.exists(root_path):
        print(f"[ERROR] Path does not exist: {root_path}")
        return selected_all_names, output_p, grip_posi_num, grip_vel_num

    materials = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    print(f"[INFO] 本次读取的材料目录: {', '.join(sorted(materials))}")

    for mat in materials:
        mat_path = os.path.join(root_path, mat)
        trials = [d for d in os.listdir(mat_path) if os.path.isdir(os.path.join(mat_path, d))]

        for trial in trials:
            trial_path = os.path.join(mat_path, trial)
            images_dir = os.path.join(trial_path, "images")

            if not os.path.exists(images_dir):
                continue

            target_val = parse_trial_target_mm(trial)
            if target_val is None:
                print(f"[WARN] Skipping folder {trial}: cannot parse target gp(mm)")
                continue

            image_files = sorted([
                f for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            for img_name in image_files:
                current_pos = parse_image_gp_mm(img_name)
                if current_pos is None:
                    continue

                full_img_path = os.path.join(images_dir, img_name)

                selected_all_names.append(full_img_path)
                grip_posi_num.append(current_pos)
                output_p.append(target_val)
                grip_vel_num.append(random.uniform(-1.0, 1.0))

    return selected_all_names, output_p, grip_posi_num, grip_vel_num


def train(model, device, train_loader, optimizer, epoch):
    cnn_encoder, mpc_layer = model
    cnn_encoder.train()
    mpc_layer.train()

    n_count = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        gripper_p = x[1][0].to(device)
        gripper_v = x[1][1].to(device)
        x_img = x[0].to(device)
        y = y.to(device).view(-1, )

        n_count += x_img.size(0)
        optimizer.zero_grad()

        output = mpc_layer(cnn_encoder(x_img), gripper_p, gripper_v)

        y = y.unsqueeze(1).expand(x_img.size(0), output.size(1))
        final_y = y[:, (output.size(1) - 1)] * 3
        final_output = output[:, (output.size(1) - 1)] * 3

        loss = F.mse_loss(output, y.float()) + F.mse_loss(final_y, final_output)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 1 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    n_count,
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                )
            )


def validation(model, device, test_loader):
    cnn_encoder, mpc_layer = model
    cnn_encoder.eval()
    mpc_layer.eval()

    loss_list = []
    with torch.no_grad():
        for x, y in test_loader:
            gripper_p = x[1][0].to(device)
            gripper_v = x[1][1].to(device)
            x_img = x[0].to(device)
            y = y.to(device).view(-1, )

            output = mpc_layer(cnn_encoder(x_img), gripper_p, gripper_v)
            y = y.unsqueeze(1).expand(x_img.size(0), output.size(1))
            final_y = y[:, (output.size(1) - 1)] * 3
            final_output = output[:, (output.size(1) - 1)] * 3
            loss = F.mse_loss(output, y.float()) + F.mse_loss(final_y, final_output)
            loss_list.append(loss.item())

    test_loss = float(np.mean(loss_list))
    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))
    return test_loss


def build_model(device):
    cnn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)
    mpc_layer = MPClayer(nHidden=CNN_embed_dim, eps=eps, nStep=nStep, del_t=del_t).to(device)

    letac_params = (
        list(cnn_encoder.fc1.parameters())
        + list(cnn_encoder.bn1.parameters())
        + list(cnn_encoder.fc2.parameters())
        + list(cnn_encoder.bn2.parameters())
        + list(cnn_encoder.fc3.parameters())
        + list(mpc_layer.parameters())
    )
    optimizer = torch.optim.Adam(letac_params, lr=learning_rate)
    return cnn_encoder, mpc_layer, optimizer


def find_latest_checkpoint(base_dir):
    candidates = []
    main_ckpt = os.path.join(base_dir, "my_letac_checkpoint.pt")
    if os.path.exists(main_ckpt):
        candidates.append(main_ckpt)

    for d in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, d)
        if os.path.isdir(run_dir) and d.startswith("run_"):
            run_ckpt = os.path.join(run_dir, "my_letac_checkpoint.pt")
            if os.path.exists(run_ckpt):
                candidates.append(run_ckpt)

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
    } if use_cuda else {
        "batch_size": batch_size,
        "shuffle": True,
    }

    print("Reading dataset from:", data_path)
    selected_all_names_, output_p_, grip_posi_num_, grip_vel_num_ = read_admittance2_data(data_path)

    if len(selected_all_names_) == 0:
        print("No data found! Check path:", os.path.abspath(data_path))
        sys.exit(1)

    print(f"Total frames loaded: {len(selected_all_names_)}")

    pv_pair_list = zip(grip_posi_num_, grip_vel_num_)
    frame_pair_list = zip(selected_all_names_, [0] * len(selected_all_names_))
    all_x_list = list(zip(frame_pair_list, pv_pair_list))
    all_y_list = output_p_

    train_list, test_list, train_label, test_label = train_test_split(
        all_x_list, all_y_list, test_size=0.2, random_state=42
    )

    train_set = Dataset_LeTac(train_list, train_label, np.arange(1, 10, 1).tolist(), transform=transform)
    valid_set = Dataset_LeTac(test_list, test_label, np.arange(1, 10, 1).tolist(), transform=transform)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    cnn_encoder, mpc_layer, optimizer = build_model(device)

    base_ckpt_dir = "checkpoints"
    os.makedirs(base_ckpt_dir, exist_ok=True)

    force_new_run = args.new_run
    latest_ckpt = None if force_new_run else find_latest_checkpoint(base_ckpt_dir)

    if latest_ckpt is not None:
        save_dir = os.path.dirname(latest_ckpt)
        ckpt_path = latest_ckpt
        best_ckpt_path = os.path.join(save_dir, "my_letac_best.pt")
        print(f"[INFO] 检测到已有断点，自动加载: {ckpt_path}")
    else:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_ckpt_dir, f"run_{now}")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "my_letac_checkpoint.pt")
        best_ckpt_path = os.path.join(save_dir, "my_letac_best.pt")
        print(f"[INFO] 新训练将保存在: {save_dir}")

    print(f"[INFO] 本次训练保存目录: {save_dir}")

    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(ckpt_path) and latest_ckpt is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        cnn_encoder.load_state_dict(checkpoint["cnn_state_dict"])
        mpc_layer.load_state_dict(checkpoint["mpc_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1

        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device)
            best_val_loss = best_ckpt.get("best_val_loss", float("inf"))

    if start_epoch >= epochs:
        print(f"[INFO] 已达到训练轮数上限 ({epochs})，将开启新训练。")
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_ckpt_dir, f"run_{now}")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "my_letac_checkpoint.pt")
        best_ckpt_path = os.path.join(save_dir, "my_letac_best.pt")
        start_epoch = 0
        best_val_loss = float("inf")
        cnn_encoder, mpc_layer, optimizer = build_model(device)
        print(f"[INFO] 本次训练保存目录: {save_dir}")

    print(f"[INFO] 从 epoch {start_epoch} 开始训练。")

    for epoch in range(start_epoch, epochs):
        val_loss = validation([cnn_encoder, mpc_layer], device, valid_loader)
        train([cnn_encoder, mpc_layer], device, train_loader, optimizer, epoch)

        torch.save(
            {
                "epoch": epoch,
                "cnn_state_dict": cnn_encoder.state_dict(),
                "mpc_state_dict": mpc_layer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            ckpt_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "cnn_state_dict": cnn_encoder.state_dict(),
                    "mpc_state_dict": mpc_layer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                best_ckpt_path,
            )
            print(f"[INFO] Best model updated at epoch {epoch}, val_loss={val_loss:.4f}")
 