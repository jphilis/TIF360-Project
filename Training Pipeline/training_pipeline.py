import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import os
import transformers
from pathlib import Path


class AudioDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            directory (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = []
        self.labels = []

        # Walk through the root directory to get subdirectories
        for index, (dirpath, dirnames, filenames) in enumerate(os.walk(root_dir)):
            # Ignore the root directory, only process subdirectories
            if dirpath != root_dir:
                for filename in filenames:
                    if filename.endswith(".wav"):
                        self.filenames.append(os.path.join(dirpath, filename))
                        self.labels.append(
                            index - 1
                        )  # index - 1 to start labeling from 0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_path = self.filenames[idx]
        label = self.labels[idx]
        # Load and preprocess the audio file
        y, sr = torchaudio.load(audio_path)  # Load audio file
        if self.transform:
            y = self.transform(y)
        return y, label


# Create the dataset
script_path = Path(__file__).resolve().parent
data_path = script_path.parent.parent / "dataset" / "training_data"
# destination_folder = script_directory / 'training_data'
dataset = AudioDataSet(data_path)
num_classes = len(set(dataset.labels))

total_size = len(dataset)
train_size = int(total_size * 0.7)
test_size = int(total_size * 0.15)
validate_size = total_size - train_size - test_size

# Split the dataset
train_dataset, validate_dataset, test_dataset = random_split(
    dataset, [train_size, validate_size, test_size]
)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load the configuration of the pre-trained model
config = transformers.AutoConfig.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
config.num_labels = num_classes  # Update the number of labels

# Load the model with the updated configuration
feature_extractor = transformers.ASTFeatureExtractor(
    sampling_rate=300000, mean=0, std=1, max_length=200, num_mel_bins=128
)
model = transformers.AutoModelForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    config=config,
    ignore_mismatched_sizes=True,
)

# Freeze encoder layers
for param in model.audio_spectrogram_transformer.encoder.parameters():
    param.requires_grad = False

# Make classifier trainable
for param in model.classifier.parameters():
    param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)


# Iterate over the dataloader
num_epochs = 50
print("Starting training...")
for epochs in range(num_epochs):
    print("training_loader.size", len(train_loader))
    for i, (batch, labels) in enumerate(train_loader):
        # batch is 4x1x600000
        # Select the first sample from the batch
        sample = feature_extractor(
            batch.squeeze().numpy(), sampling_rate=300000, return_tensors="pt"
        )
        input = F.interpolate(
            sample["input_values"].unsqueeze(0),
            size=(1024, 128),
            mode="bilinear",
            align_corners=False,
        )
        input = input.squeeze(0)
        logits = model(input).logits
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(
                "Epoch "
                + str(epochs + 1)
                + ", iteration "
                + str(i)
                + ": "
                + str(loss.item())
            )
    print("Epoch " + str(epochs + 1) + " completed.")

    total_correct = 0
    total_samples = 0
    for i, (batch, labels) in enumerate(validate_loader):
        sample = feature_extractor(
            batch.squeeze().numpy(), sampling_rate=300000, return_tensors="pt"
        )
        input = F.interpolate(
            sample["input_values"].unsqueeze(0),
            size=(1024, 128),
            mode="bilinear",
            align_corners=False,
        )
        print("interpolated")
        input = input.squeeze(0)
        outputs = model(input).logits
        print("outputs")
        _, predicted_labels = torch.max(outputs, 1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    message = (
        "Validation accuracy after epoch " + str(epochs + 1) + ": " + str(accuracy)
    )
    print(message)

total_correct = 0
total_samples = 0
for i, (batch, labels) in enumerate(test_loader):
    sample = feature_extractor(
        batch.squeeze().numpy(), sampling_rate=300000, return_tensors="pt"
    )
    input = F.interpolate(
        sample["input_values"].unsqueeze(0),
        size=(1024, 128),
        mode="bilinear",
        align_corners=False,
    )
    input = input.squeeze(0)
    outputs = model(input).logits
    _, predicted_labels = torch.max(outputs, 1)
    total_correct += (predicted_labels == labels).sum().item()
    total_samples += labels.size(0)
    accuracy = total_correct / total_samples

accuracy = total_correct / total_samples
message = "Final test accuracy: " + str(accuracy)
print(message)
