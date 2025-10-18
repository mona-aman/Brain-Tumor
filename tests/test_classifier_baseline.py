# test_classifier_baseline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from data.dataset import MRIDataset, get_test_transforms
from models.classifier import BrainTumorClassifier

# Load classifier
classifier = BrainTumorClassifier(num_classes=4)
checkpoint = torch.load('results/models/checkpoint_best.pth', map_location='cuda')
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier = classifier.cuda().eval()

# Load test data
test_dataset = MRIDataset(
    root_dir='data/primary',
    split='Testing',
    transform=get_test_transforms(128)
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Test on original images
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        
        if (labels == -1).any():
            continue
        
        outputs = classifier(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100. * correct / total
print(f"Classifier accuracy on original images: {accuracy:.2f}%")
print(f"Expected: ~94.89%")

if accuracy < 80:
    print("❌ PROBLEM: Classifier is not working correctly!")
else:
    print("✓ Classifier works fine on original images")