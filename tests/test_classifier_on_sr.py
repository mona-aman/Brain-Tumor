# test_classifier_on_sr.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.esrgan import Generator
from models.classifier import BrainTumorClassifier
from data.dataset import MRIDataset, get_test_transforms
from torch.utils.data import DataLoader

# Load models
generator = Generator().cuda()
gen_checkpoint = torch.load('results/checkpoints/esrgan_best.pth')
generator.load_state_dict(gen_checkpoint['generator_state_dict'])
generator.eval()

classifier = BrainTumorClassifier(num_classes=4).cuda()
clf_checkpoint = torch.load('results/models/checkpoint_best.pth')
classifier.load_state_dict(clf_checkpoint['model_state_dict'])
classifier.eval()

# Load data
test_dataset = MRIDataset('data/primary', 'Testing', get_test_transforms(128))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

correct_hr = 0
correct_sr = 0
total = 0

with torch.no_grad():
    for hr_img, labels in test_loader:
        hr_img = hr_img.cuda()
        labels = labels.cuda()
        
        if (labels == -1).any():
            continue
        
        # Test on HR (original)
        outputs_hr = classifier(hr_img)
        _, pred_hr = outputs_hr.max(1)
        correct_hr += pred_hr.eq(labels).sum().item()
        
        # Generate SR
        lr_img = F.interpolate(hr_img, scale_factor=0.25, mode='bilinear')
        sr_img = generator(lr_img, target_size=hr_img.shape[2:])
        
        # Test on SR
        outputs_sr = classifier(sr_img)
        _, pred_sr = outputs_sr.max(1)
        correct_sr += pred_sr.eq(labels).sum().item()
        
        total += labels.size(0)

print(f"Classifier accuracy on HR images: {100.*correct_hr/total:.2f}%")
print(f"Classifier accuracy on SR images: {100.*correct_sr/total:.2f}%")

if 100.*correct_sr/total < 50:
    print("\n❌ PROBLEM: SR images are not being classified correctly!")
    print("Possible issues:")
    print("  1. Normalization mismatch")
    print("  2. SR images are too distorted")
    print("  3. Image size mismatch")