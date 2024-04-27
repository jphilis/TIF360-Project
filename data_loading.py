import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import transformers


# Define transformations
transform = transforms.Compose([
    transforms.Resize((109*16, 5*16+4)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Create the dataset
script_path = os.path.abspath(__file__)
data_path = os.path.join(os.path.dirname(script_path), 'data/spectrograms')
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over the dataloader
for i, (batch, labels) in enumerate(dataloader):
    print(f'Batch {i}: {batch.shape}, Labels: {labels}')
    batch2 = batch
    if i == 0:
        break

# Load the pre-trained model
model = transformers.ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

print(model.get_input_embeddings())

# Select the first sample from the batch
sample = batch2[0]

# Remove the extra dimension
sample = sample.squeeze(0)

# Pass the sample to the model
predictions = model(sample.unsqueeze(0))

# Print the predictions

print(predictions)



