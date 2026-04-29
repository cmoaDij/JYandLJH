import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import random

# Add parent directory to path to import functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.functions import *

parser = argparse.ArgumentParser(description="Train LeTac-MPC with admittance dataset")
group = parser.add_mutually_exclusive_group()
group.add_argument("--new-run", action="store_true", help="Force a new training run (ignore checkpoints)")
group.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
args = parser.parse_args()

# EncoderCNN architecture
CNN_hidden1, CNN_hidden2 = 128, 128 
CNN_embed_dim = 20  
res_size = 224       
dropout_p = 0.15  

# Training parameters
epochs = 2000
batch_size = 1024
learning_rate = 1e-4
eps = 1e-4
nStep = 15
del_t = 1/25

# Points to your new dataset root
data_path = "dataset_dealed"

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])


# Data Scaling Parameters
# Original data was roughly 0-30. User data is 0-1000.
# We scale down by 33.3 (approx 1000/30) to match the model's expected input range.
SCALE_FACTOR = 33.33 

def read_admittance_data(root_path):
    """
    Reads data from the admittance dataset structure WITHOUT CSV:
    Root/
      Material/ (e.g. Black, Red)
        trial_XXX_gp_YYY/ (YYY is target position)
          images/
            frame_ZZZ_gp_PPP.jpg (PPP is current gripper pos)
    """
    selected_all_names = []  # Image paths
    output_p = []            # Trial target (final gripper pos)
    grip_posi_num = []       # Current gripper pos
    grip_vel_num = []        # Current gripper vel (randomly generated)
    
    # Traverse materials (Black, Red, etc.)
    materials = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    print(f"[INFO] 本次读取的材料目录: {', '.join(sorted(materials))}")
    
    for mat in materials:
        mat_path = os.path.join(root_path, mat)
        trials = [d for d in os.listdir(mat_path) if os.path.isdir(os.path.join(mat_path, d))]
        
        for trial in trials:
            trial_path = os.path.join(mat_path, trial)
            images_dir = os.path.join(trial_path, 'images')
            
            if not os.path.exists(images_dir):
                continue
                
            # 1. Parse Target Value from folder name: trial_XXX_gp_YYY
            # YYY is the target gripper position (e.g., 123)
            try:
                gp_index = trial.find('_gp_')
                if gp_index != -1:
                    target_val = float(trial[gp_index + 4:])
                else:
                    print(f"Skipping folder {trial}: cannot parse target gp")
                    continue
            except ValueError:
                print(f"Skipping folder {trial}: invalid target gp value")
                continue

            # 2. Iterate over images in the 'images' folder
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
            
            for img_name in image_files:
                # 3. Parse Current Gripper Position from filename: frame_ZZZ_gp_PPP.jpg
                # PPP is the current gripper position (e.g., 4)
                try:
                    # Find last occurrence of '_gp_' to define start of position
                    # And '.jpg' to define end
                    gp_start = img_name.rfind('_gp_')
                    ext_start = img_name.rfind('.jpg')
                    
                    if gp_start != -1 and ext_start != -1:
                        current_pos = float(img_name[gp_start + 4 : ext_start])
                    else:
                        continue # Skip images that don't match format
                except ValueError:
                    continue

                full_img_path = os.path.join(images_dir, img_name)
                
                selected_all_names.append(full_img_path)
                
                # Apply Scaling during data loading
                grip_posi_num.append(current_pos / SCALE_FACTOR)
                output_p.append(target_val / SCALE_FACTOR)
                
                # 4. Generate Random Velocity: uniform noise in [-1, 1]
                grip_vel_num.append((random.random() - 0.5) * 2.0)

    return selected_all_names, output_p, grip_posi_num, grip_vel_num

def train(model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, MPC_layer = model
    cnn_encoder.train()
    MPC_layer.train()
    losses = []

    N_count = 0 
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        # X[1][0] is gripper_p, X[1][1] is gripper_v
        gripper_p = X[1][0].to(device)
        gripper_v = X[1][1].to(device)
        
        # X[0] is image
        X_img = X[0].to(device)
        y = y.to(device).view(-1, )
        
        N_count += X_img.size(0)
        optimizer.zero_grad()
        
        output = MPC_layer(cnn_encoder(X_img), gripper_p, gripper_v)
        
        # Original loss calculation logic
        y = y.unsqueeze(1).expand(X_img.size(0), output.size(1))
        # Emphasis on the last step (index -1)
        final_y = y[:, (output.size(1)-1)] * 3
        final_output = output[:, (output.size(1)-1)] * 3
        
        loss = F.mse_loss(output, y.float()) + F.mse_loss(final_y, final_output)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % 1 == 0:  # Changed to print every batch for small datasets
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

def validation(model, device, test_loader):
    cnn_encoder, MPC_layer = model
    cnn_encoder.eval()
    MPC_layer.eval()
    loss_list = []
    with torch.no_grad():
        for X, y in test_loader:
            gripper_p = X[1][0].to(device)
            gripper_v = X[1][1].to(device)
            X_img = X[0].to(device)
            y = y.to(device).view(-1, )
            output = MPC_layer(cnn_encoder(X_img), gripper_p, gripper_v)
            y = y.unsqueeze(1).expand(X_img.size(0), output.size(1))
            final_y = y[:, (output.size(1)-1)] * 3
            final_output = output[:, (output.size(1)-1)] * 3
            loss = F.mse_loss(output, y.float()) + F.mse_loss(final_y, final_output)
            loss_list.append(loss.item())
    test_loss = float(np.mean(loss_list))
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss

# --- Main Execution ---

use_cuda = torch.cuda.is_available()                  
device = torch.device("cuda" if use_cuda else "cpu")   

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16, 'pin_memory': True} if use_cuda else {}

print("Reading dataset from:", data_path)
selected_all_names_, output_p_, grip_posi_num_, grip_vel_num_ = read_admittance_data(data_path)

if len(selected_all_names_) == 0:
    print("No data found! Check path:", os.path.abspath(data_path))
    sys.exit(1)

print(f"Total frames loaded: {len(selected_all_names_)}")

# Prepare data for DataLoader
# Using zip to pair gripper_pos and gripper_vel as the second input element
pv_pair_list = zip(grip_posi_num_, grip_vel_num_)
# Using zip to pair image path and a dummy index (original code used index for something, but Dataset_LeTac handles it)
# Note: Dataset_LeTac expects folders_pv_pair = ((image_path, index), (pos, vel))
# But original Dataset_LeTac implementation: 
#   folders = list(np.array(tuple(folders_pv_pair),dtype=object)[:,0])
#   pv_pairs = list(np.array(tuple(folders_pv_pair),dtype=object)[:,1])
#   ...
#   folder = self.folders[index]  <- this is the image path list passed in
#   image = Image.open(selected_folder[0]) 

# Let's align exactly with Dataset_LeTac expectation: 
# It expects the first element of the pair to be a LIST/TUPLE where index 0 is the image path.
frame_pair_list = zip(selected_all_names_, [0]*len(selected_all_names_)) 
all_x_list = list(zip(frame_pair_list, pv_pair_list))        
all_y_list = output_p_

# Train/Test split
train_list, test_list, train_label, test_label = train_test_split(all_x_list, all_y_list, test_size=0.2, random_state=42)

# Create Datasets
# Note: passing np.arange(1, 10, 1).tolist() as 'frames' arg, though it might not be used if we provide direct paths
train_set = Dataset_LeTac(train_list, train_label, np.arange(1, 10, 1).tolist(), transform=transform)
valid_set = Dataset_LeTac(test_list, test_label, np.arange(1, 10, 1).tolist(), transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

def build_model(device):
    cnn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(device)
    MPC_layer = MPClayer(nHidden=CNN_embed_dim, eps=eps, nStep=nStep, del_t=del_t).to(device)
    letac_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                   list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                   list(cnn_encoder.fc3.parameters()) + list(MPC_layer.parameters())
    optimizer = torch.optim.Adam(letac_params, lr=learning_rate)
    return cnn_encoder, MPC_layer, optimizer

# Create model
cnn_encoder, MPC_layer, optimizer = build_model(device)


# === Checkpoint/Resume/Best Model Logic ===
import datetime

# 是否强制开启新训练（默认 False：优先断点续训）
FORCE_NEW_RUN = args.new_run

base_ckpt_dir = "checkpoints"
os.makedirs(base_ckpt_dir, exist_ok=True)

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

latest_ckpt = None if FORCE_NEW_RUN else find_latest_checkpoint(base_ckpt_dir)

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

# 2. 支持断点续训
start_epoch = 0
best_val_loss = float('inf')
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    cnn_encoder.load_state_dict(checkpoint['cnn_state_dict'])
    MPC_layer.load_state_dict(checkpoint['mpc_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        best_val_loss = best_ckpt.get('best_val_loss', float('inf'))

if start_epoch >= epochs:
    print(f"[INFO] 已达到训练轮数上限 ({epochs})，将开启新训练。")
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_ckpt_dir, f"run_{now}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "my_letac_checkpoint.pt")
    best_ckpt_path = os.path.join(save_dir, "my_letac_best.pt")
    start_epoch = 0
    best_val_loss = float('inf')
    cnn_encoder, MPC_layer, optimizer = build_model(device)
    print(f"[INFO] 本次训练保存目录: {save_dir}")

print(f"[INFO] 从 epoch {start_epoch} 开始训练。")


# 3. 训练主循环，只保存最新和best模型
for epoch in range(start_epoch, epochs):
    # 验证，获取当前val loss
    val_loss = validation([cnn_encoder, MPC_layer], device, valid_loader)
    # 训练
    train([cnn_encoder, MPC_layer], device, train_loader, optimizer, epoch)

    # 保存最新模型
    torch.save({
        'epoch': epoch,
        'cnn_state_dict': cnn_encoder.state_dict(),
        'mpc_state_dict': MPC_layer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, ckpt_path)

    # 保存best模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn_encoder.state_dict(),
            'mpc_state_dict': MPC_layer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, best_ckpt_path)
        print(f"[INFO] Best model updated at epoch {epoch}, val_loss={val_loss:.4f}")
