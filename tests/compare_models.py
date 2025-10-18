# compare_models.py
import torch

standard = torch.load('results/checkpoints/esrgan_best.pth')
taskaware = torch.load('results/checkpoints/taskaware_best.pth')

print("Standard - Epoch:", standard.get('epoch'))
print("Task-Aware - Epoch:", taskaware.get('epoch'))

print("\nStandard metrics:", standard.get('metrics'))
print("Task-Aware metrics:", taskaware.get('metrics'))

# Check if weights are actually different
std_weights = standard['generator_state_dict']['initial_conv.weight']
task_weights = taskaware['generator_state_dict']['initial_conv.weight']

difference = (std_weights - task_weights).abs().mean()
print(f"\nWeight difference: {difference:.6f}")
if difference < 0.001:
    print("⚠️ Models are almost identical - training might have failed!")
else:
    print("✓ Models are different")
