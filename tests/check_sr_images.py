# check_sr_images.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.esrgan import Generator

# Load generator
generator = Generator().cuda()
checkpoint = torch.load('results/checkpoints/esrgan_best.pth')
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Generate one SR image
from data.dataset import MRIDataset, get_test_transforms
test_dataset = MRIDataset('data/primary', 'Testing', get_test_transforms(128))
hr_img, label = test_dataset[0]
hr_img = hr_img.unsqueeze(0).cuda()

# Create LR and generate SR
lr_img = F.interpolate(hr_img, scale_factor=0.25, mode='bilinear')
with torch.no_grad():
    sr_img = generator(lr_img, target_size=hr_img.shape[2:])

print("=== Image Statistics ===")
print(f"Original HR: min={hr_img.min():.3f}, max={hr_img.max():.3f}, mean={hr_img.mean():.3f}")
print(f"LR input:    min={lr_img.min():.3f}, max={lr_img.max():.3f}, mean={lr_img.mean():.3f}")
print(f"SR output:   min={sr_img.min():.3f}, max={sr_img.max():.3f}, mean={sr_img.mean():.3f}")

print("\n=== Expected for Classifier ===")
print("Should be: mean ~0.0, range [-1, 1] OR mean ~0.5, range [0, 1]")
print("Depends on normalization used in training")

# Save for visual inspection
import torchvision
torchvision.utils.save_image(hr_img, 'debug_hr.png')
torchvision.utils.save_image(lr_img, 'debug_lr.png')
torchvision.utils.save_image(sr_img, 'debug_sr.png')
print("\nSaved debug images: debug_hr.png, debug_lr.png, debug_sr.png")
print("Open these to see if SR image looks reasonable")