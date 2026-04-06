# generate_model.py
import torch
from CNN import CNN

# Number of classes from your mapping
num_classes = 39

# Initialize model
model = CNN(num_classes)

# Save dummy (untrained) model
torch.save(model.state_dict(), "plant_disease_model_1_latest.pt")

print("✅ Dummy model file 'plant_disease_model_1_latest.pt' created successfully.")
