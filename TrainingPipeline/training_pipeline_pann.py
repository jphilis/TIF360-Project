from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torchaudio.transforms as ta_transforms
import torchvision.transforms.v2 as tv_transforms
import datetime
import random
import sys

# Adjusting paths and importing batdetect2
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
batdetect2_main_dir = os.path.join(parent_dir, "batdetect2-main")
sys.path.append(batdetect2_main_dir)
print("batdetect2_main_dir", batdetect2_main_dir)
from batdetect2 import api

# Check for wandb
try:
    import wandb
    use_wandb = True
except ImportError:
    use_wandb = False

# Define AudioDataSet class
class AudioDataSet(Dataset):
    def __init__(self, root_dir, file_count_threshold, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.file_count_threshold = file_count_threshold

        for index, (dirpath, dirnames, filenames) in enumerate(os.walk(root_dir)):
            if dirpath != root_dir:
                class_file_names = [os.path.join(dirpath, f) for f in filenames if f.endswith(".wav")]
                selected_filenames = random.choices(class_file_names, k=min(self.file_count_threshold, len(class_file_names)))
                selected_labels = [index - 1] * len(selected_filenames)
                self.filenames.extend(selected_filenames)
                self.labels.extend(selected_labels)

        if transform is None:
            self.transform = tv_transforms.Compose([
                ta_transforms.MelSpectrogram(
                    sample_rate=256000,
                    n_fft=512,
                    n_mels=128,
                    f_max=120000,
                    f_min=20000,
                    normalized=False,
                    win_length=512
                ),
                tv_transforms.Lambda(lambda img: img.transpose(1, 2)),
                tv_transforms.Resize((1024, 128))
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_path = self.filenames[idx]
        label = self.labels[idx]
        audio = api.load_audio(audio_path, target_samp_rate=256000)
        spec = api.generate_spectrogram(audio)
        spec = torch.transpose(spec, 2, 3)
        y = F.interpolate(spec, size=(1024, 128), mode="bilinear")
        cmap = plt.get_cmap("viridis")
        S_color = cmap(y.cpu().numpy()).squeeze()
        if S_color.shape[2] == 4:
            S_color = S_color[..., :3]
        image = torch.tensor(S_color).permute(2, 0, 1).float()
        return image, label

# Define the model class
class Wavegram_Logmel_Cnn14Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrained_model = torch.hub.load('qiuqiangkong/panns_inference', 'Cnn14', pretrained=True, force_reload=True)
        self.pretrained_model.fc = torch.nn.Linear(self.pretrained_model.fc.in_features, num_classes)

    def forward(self, input):
        x = self.pretrained_model(input)
        return x

# Main function
def main():
    script_path = Path(__file__).resolve().parent
    data_path = script_path.parent.parent / "dataset" / "training_data_100ms_noise_50_2"

    train_dataset = AudioDataSet(data_path / "train", 3000)
    validate_dataset = AudioDataSet(data_path / "validate", 300)
    test_dataset = AudioDataSet(data_path / "test", 20)

    num_classes = len(os.listdir(data_path / "train"))
    class_names = [p.stem for p in Path(data_path / "train").glob("*")]
    print("Number of classes:", num_classes)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = Wavegram_Logmel_Cnn14Model(num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if use_wandb:
        wandb.init(project="TIF360", entity="jonatca")

    num_epochs = 50
    print("Starting training...")
    best_loss = 1000
    best_accuracy = 0
    for epoch in range(num_epochs):
        print("training_loader.size", len(train_loader))
        total_loss = 0
        model.train()
        for i, (batch, labels) in enumerate(train_loader):
            batch, labels = batch.to(device), labels.to(device)
            if len(batch.size()) != 4:
                print("something wrong with dimensions here")
                print("batch.size", batch.size())
                continue
            logits = model(batch)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                try:
                    print(f"Epoch {epoch + 1}, iteration {i}: {loss.item()}")
                except Exception as e:
                    print("Error: ", e)
        avg_loss = total_loss / len(train_loader)
        if use_wandb:
            wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})
        
        model.eval()
        total_correct = 0
        total_samples = 0
        tot_val_loss = 0
        with torch.no_grad():
            for i, (batch, labels) in enumerate(validate_loader):
                batch, labels = batch.to(device), labels.to(device)
                if len(batch.size()) != 4:
                    print("WARNING something wrong with dimensions here")
                    print("batch.size", batch.size())
                    continue
                outputs = model(batch)
                _, predicted_labels = torch.max(outputs, 1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)
                loss = criterion(outputs, labels)
                tot_val_loss += loss.item()
                if i % 1000 == 0:
                    time_of_day = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"{time_of_day} Epoch {epoch + 1}, iteration {i}: {loss.item()}")
        accuracy = total_correct / total_samples
        val_loss = tot_val_loss / len(validate_loader)
        if use_wandb:
            wandb.log({"validation_accuracy": accuracy, "epoch": epoch + 1, "validation_loss": val_loss})
        
        message = f"Validation accuracy after epoch {epoch + 1}: {accuracy} loss: {val_loss}"
        if val_loss < best_loss or accuracy > best_accuracy:
            best_loss = val_loss
            best_accuracy = accuracy
            folder_name = "cnn"
            os.makedirs(folder_name, exist_ok=True)
            torch.save(model.state_dict(), f"{folder_name}/best_model_loss_cnn_wavegram_logmel_cnn14_loss{val_loss}_acc{accuracy}.pth")
            message += " (model saved)"
        print(message)

    model.load_state_dict(torch.load(f"{folder_name}/best_model_loss_cnn_wavegram_logmel_cnn14_loss{best_loss}_acc{best_accuracy}.pth"))
    predicted = []
    actual = []

    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for i, (batch, labels) in enumerate(test_loader):
            actual.extend(labels.cpu().numpy())
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch)
            _, predicted_labels = torch.max(outputs, 1)
            predicted.extend(predicted_labels.cpu().numpy())
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    np.save("test_labels.npy", actual)
    np.save("test_predictions.npy", predicted)

    accuracy = total_correct / total_samples
    message = "Final test accuracy: " + str(accuracy)
    print(message)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(actual, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()
