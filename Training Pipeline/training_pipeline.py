import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import transformers

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
                    if filename.endswith('.wav'):
                        self.filenames.append(os.path.join(dirpath, filename))
                        self.labels.append(index - 1)  # index - 1 to start labeling from 0

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
script_path = os.path.abspath(__file__)
data_path = os.path.join(os.path.dirname(script_path), 'training_data')
dataset = AudioDataSet(data_path)
num_classes = len(set(dataset.labels))

# Load the configuration of the pre-trained model
config = transformers.AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
config.num_labels = num_classes  # Update the number of labels

# Load the model with the updated configuration
model = transformers.AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=config, ignore_mismatched_sizes=True)

# Freeze encoder layers
for param in model.audio_spectrogram_transformer.encoder.parameters():
    param.requires_grad = False

# Make classifier trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)


# Iterate over the dataloader
for i, (batch, labels) in enumerate(dataloader):
    # Load the pre-trained model
    feature_extractor = transformers.ASTFeatureExtractor(sampling_rate=300000, mean=0, std=1, max_length=200, num_mel_bins=128)

    # Select the first sample from the batch
    sample = feature_extractor(batch.squeeze().numpy(), sampling_rate=300000, return_tensors='pt')
    input = F.interpolate(sample['input_values'].unsqueeze(0), size=(1024, 128), mode='bilinear', align_corners=False)
    input = input.squeeze(0)
    logits = model(input).logits
    predicted_class_ids = torch.argmax(logits, dim=-1)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    if loss.requires_grad:
            loss.backward()
            optimizer.step()