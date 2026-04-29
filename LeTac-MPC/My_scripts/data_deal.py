import os
import shutil
from pathlib import Path

def process_dataset(source_root, target_root, gp_interval=10):
    """
    Process dataset by selecting images with gripper position (gp) intervals.
    
    Args:
        source_root: Path to dataset_admittance
        target_root: Path to dataset_dealed
        gp_interval: The interval between selected gripper positions (default: 10)
    """
    
    if not os.path.exists(source_root):
        print(f"[ERROR] Source path does not exist: {source_root}")
        return
    
    # Create target root directory
    os.makedirs(target_root, exist_ok=True)
    
    # Traverse materials (Black, Red, Empty, etc.)
    materials = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    print(f"[INFO] Processing materials: {', '.join(sorted(materials))}")
    
    total_original = 0
    total_selected = 0
    
    for mat in materials:
        mat_source_path = os.path.join(source_root, mat)
        mat_target_path = os.path.join(target_root, mat)
        os.makedirs(mat_target_path, exist_ok=True)
        
        trials = [d for d in os.listdir(mat_source_path) if os.path.isdir(os.path.join(mat_source_path, d))]
        
        for trial in trials:
            trial_source_path = os.path.join(mat_source_path, trial)
            trial_target_path = os.path.join(mat_target_path, trial)
            
            images_source_dir = os.path.join(trial_source_path, 'images')
            
            if not os.path.exists(images_source_dir):
                print(f"[WARN] No images folder in: {trial_source_path}")
                continue
            
            # Create target trial directory structure
            os.makedirs(trial_target_path, exist_ok=True)
            images_target_dir = os.path.join(trial_target_path, 'images')
            os.makedirs(images_target_dir, exist_ok=True)
            
            # Copy data.csv if it exists
            data_csv_source = os.path.join(trial_source_path, 'data.csv')
            if os.path.exists(data_csv_source):
                data_csv_target = os.path.join(trial_target_path, 'data.csv')
                shutil.copy2(data_csv_source, data_csv_target)
            
            # Get all image files and sort them
            image_files = sorted([f for f in os.listdir(images_source_dir) if f.endswith('.jpg')])
            
            if len(image_files) == 0:
                print(f"[WARN] No images found in: {images_source_dir}")
                continue
            
            total_original += len(image_files)
            
            # Parse gp values and create a mapping: gp -> filename
            gp_to_file = {}
            for img_name in image_files:
                try:
                    # Parse gripper position from filename: frame_ZZZ_gp_PPP.jpg
                    gp_start = img_name.rfind('_gp_')
                    ext_start = img_name.rfind('.jpg')
                    
                    if gp_start != -1 and ext_start != -1:
                        gp_value = float(img_name[gp_start + 4 : ext_start])
                        gp_to_file[gp_value] = img_name
                except ValueError:
                    continue
            
            if len(gp_to_file) == 0:
                print(f"[WARN] No valid gp values parsed in: {images_source_dir}")
                continue
            
            # Sort gp values
            sorted_gp_values = sorted(gp_to_file.keys())
            
            # Find the starting gp (first image's gp)
            start_gp = sorted_gp_values[0]
            
            # Select images with gp intervals
            selected_count = 0
            current_target_gp = start_gp
            
            while current_target_gp <= sorted_gp_values[-1]:
                # Find the closest gp to current_target_gp
                closest_gp = min(sorted_gp_values, key=lambda x: abs(x - current_target_gp))
                
                # Only select if the difference is within tolerance (e.g., ±0.5)
                if abs(closest_gp - current_target_gp) <= 0.5:
                    img_name = gp_to_file[closest_gp]
                    source_img_path = os.path.join(images_source_dir, img_name)
                    target_img_path = os.path.join(images_target_dir, img_name)
                    
                    # Copy the image
                    shutil.copy2(source_img_path, target_img_path)
                    selected_count += 1
                
                # Move to next target gp
                current_target_gp += gp_interval
            
            total_selected += selected_count
            print(f"[INFO] {mat}/{trial}: {len(image_files)} -> {selected_count} images")
    
    print(f"\n[SUMMARY]")
    print(f"Total original images: {total_original}")
    print(f"Total selected images: {total_selected}")
    print(f"Selection ratio: {total_selected/total_original*100:.1f}%")
    print(f"Output directory: {os.path.abspath(target_root)}")

if __name__ == '__main__':
    # Define paths
    source_dataset = "dataset_admittance"
    target_dataset = "dataset_dealed"
    
    # Process with gp interval of 10
    print("=" * 60)
    print("Dataset Processing: Selecting images with gp interval = 10")
    print("=" * 60)
    
    process_dataset(source_dataset, target_dataset, gp_interval=10)
    
    print("\n[DONE] Processing complete!")
