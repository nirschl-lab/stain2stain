import os
import numpy as np
import random
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import pdb

from src.data.paired_data_module import PairedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
# ckpt_path = "/home/wisc/maheswararao/code/stain2stain/stain2stain/artifacts/model-leaf61cl:v2/model.ckpt"
ckpt_path = "logs/train/runs/2026-01-28_10-08-19/checkpoints/best-199-0.0228.ckpt"
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
# data_dir = "/data1/shared/data/destain_restain/he_amyloid/positive_crops/flow_matching/"
# csv_file_name = "dataset_nirschl_et_al_2026_metadata.csv"
data_dir = "/data1/shared/data/destain_restain/he_lhe"
csv_file_name = "dataset_he-lhe_512x512_metadata2.csv"
save_base_path = "/data1/shared/data/destain_restain/he_lhe/inference/"
save_folder = "simple_flowmatching"
image_size = 256
use_augmentation = False
source_column = 'he_filepath'
target_column = 'target_filepath'
num_images_to_infer = 50
num_steps = 2  # Number of ODE integration steps


def denormalize(img):
    return (img * 0.5 + 0.5).clamp(0, 1)

def main():
    # Create save directory
    full_save_path = os.path.join(save_base_path, save_folder)
    os.makedirs(full_save_path, exist_ok=True)
    print(f"Save path created: {full_save_path}")
    
    # Load model from config
    print("Loading model from config...")
    model_config = OmegaConf.load("configs/model/conditional_flow_matching.yaml")
    model = hydra.utils.instantiate(model_config)
    model = model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Create dataset with return_filename enabled
    dataset = PairedDataset(
        data_dir=data_dir,
        csv_file_name=csv_file_name,
        folder="test",
        source_column=source_column,
        target_column=target_column,
        image_size=image_size,
        use_augmentation=use_augmentation,
        return_filename=True  # Enable filename return
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Process first N images
    num_to_process = min(num_images_to_infer, len(dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(num_to_process), desc="Processing images"):
            try:
                # Get batch
                source_img, target_img, source_filename, target_filename = dataset[idx]
                
                # Add batch dimension
                source_img = source_img.unsqueeze(0).to(device)
                target_img = target_img.unsqueeze(0).to(device)
                
                # Generate prediction
                generated_img = model.generate(source_img, num_steps=num_steps)
                
                # Denormalize images
                source_np = denormalize(source_img[0]).cpu().permute(1, 2, 0).numpy()
                target_np = denormalize(target_img[0]).cpu().permute(1, 2, 0).numpy()
                generated_np = denormalize(generated_img[0]).cpu().permute(1, 2, 0).numpy()
                
                # Create filename for saving
                # Extract base name without extension
                source_base = os.path.splitext(source_filename)[0]
                
                # Create a figure with three images side by side with titles
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Source image
                axes[0].imshow(source_np)
                axes[0].set_title('Source', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                # Target image
                axes[1].imshow(target_np)
                axes[1].set_title('Target', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                # Generated image
                axes[2].imshow(generated_np)
                axes[2].set_title('Generated', fontsize=14, fontweight='bold')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save combined image
                combined_name = f"{source_base}.png"
                combined_path = os.path.join(full_save_path, combined_name)
                plt.savefig(combined_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                print(f"Saved: {combined_name}")
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
    
    print(f"\nInference complete! Results saved to: {full_save_path}")
    print(f"Processed {num_to_process} images")


if __name__ == "__main__":
    main()


    
