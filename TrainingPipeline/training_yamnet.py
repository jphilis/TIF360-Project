from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
from pathlib import Path
import torchaudio.transforms as ta_transforms
import torchvision.transforms.v2 as tv_transforms
import copy
import datetime
import random
import torchvision
from PIL import Image
import sys
from efficientnet_pytorch import EfficientNet

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
batdetect2_main_dir = os.path.join(parent_dir, "batdetect2-main")
sys.path.append(batdetect2_main_dir)
print("batdetect2_main_dir", batdetect2_main_dir)
from batdetect2 import api

try:
    import wandb

    use_wandb = True
except ImportError:
    use_wandb = False

class AudioDataSet(Dataset):
    def __init__(self, root_dir, file_count_threshold, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.file_count_threshold = file_count_threshold

        for index, (dirpath, dirnames, filenames) in enumerate(os.walk(root_dir)):
            selected_filenames = []
            if dirpath != root_dir:
                class_file_names = []
                for filename in filenames:
                    if filename.endswith(".wav"):
                        class_file_names.append(os.path.join(dirpath, filename))
                selected_filenames = random.choices(
                    class_file_names,
                    k=min(self.file_count_threshold, len(class_file_names)),
                )
                selected_labels = [index - 1] * len(selected_filenames)
                self.filenames.extend(selected_filenames)
                self.labels.extend(selected_labels)

        if transform is None:
            self.transform = tv_transforms.Compose(
                [
                    ta_transforms.MelSpectrogram(
                        sample_rate=256000,
                        n_fft=512,
                        n_mels=128,
                        f_max=120000,
                        f_min=20000,
                        normalized=False,
                        win_length=512,
                    ),
                    tv_transforms.Lambda(lambda img: img.transpose(1, 2)),
                    tv_transforms.Resize((1024, 128)),
                ]
            )

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

def augment_dataset(dataset, augmentations=["Normal"]) -> ConcatDataset:
    number_of_FM = 3
    number_of_TM = 10

    spectrogramed = ta_transforms.MelSpectrogram(
        sample_rate=300000, n_mels=128, f_max=120000, f_min=20000, normalized=True
    )
    transposed = tv_transforms.Lambda(lambda img: img.transpose(1, 2))
    resized = tv_transforms.Resize((1024, 128))
    frequency_masked = ta_transforms.FrequencyMasking(freq_mask_param=10)
    time_masked = ta_transforms.TimeMasking(time_mask_param=40)

    transform = tv_transforms.Compose([spectrogramed, transposed, resized])

    transform_FM = tv_transforms.Compose(
        [spectrogramed]
        + [frequency_masked for _ in range(number_of_FM)]
        + [transposed, resized]
    )

    transform_TM = tv_transforms.Compose(
        [spectrogramed]
        + [time_masked for _ in range(number_of_TM)]
        + [transposed, resized]
    )

    transform_FTM = tv_transforms.Compose(
        [spectrogramed]
        + [frequency_masked for _ in range(number_of_FM)]
        + [time_masked for _ in range(number_of_TM)]
        + [transposed, resized]
    )

    dataset_FM = copy.deepcopy(dataset)
    dataset_TM = copy.deepcopy(dataset)
    dataset_FTM = copy.deepcopy(dataset)

    dataset_FM.transform = transform_FM
    dataset_TM.transform = transform_TM
    dataset_FTM.transform = transform_FTM

    augmentation_list = {
        "Normal": dataset,
        "FM": dataset_FM,
        "TM": dataset_TM,
        "FTM": dataset_FTM,
    }

    combined_dataset = ConcatDataset([augmentation_list[aug] for aug in augmentations])

    return combined_dataset

class CNN(torch.nn.Module):
    def __init__(self, num_classes, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.linear1 = torch.nn.Linear(1280, 512)  # Adjusted for EfficientNet output
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, num_classes)

    def forward(self, input):
        x = self.pretrained_model.extract_features(input)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x

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

    # Load the EfficientNet model
    pretrained_model = EfficientNet.from_pretrained('efficientnet-b0')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = CNN(num_classes, pretrained_model).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if use_wandb:
        wandb.init(project="TIF360", entity="jonatca")

    num_epochs = 50
    print("Starting training...")
    best_loss = 1000
    best_accuracy = 0
    for epochs in range(num_epochs):
        print("training_loader.size", len(train_loader))
        total_loss = 0
        loss = 1000
        model.train()
        for i, (batch, labels) in enumerate(train_loader):
            batch, labels = batch.to(device), labels.to(device)
            input = batch
            if len(input.size()) != 4:
                print("something wrong with dimensions here")
                print("input.size", input.size())
                continue
            logits = model(input)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                try:
                    print(
                        "Epoch "
                        + str(epochs + 1)
                        + ", iteration "
                        + str(i)
                        + ": "
                        + str(loss.item())
                    )
                except Exception as e:
                    print("Error: ", e)
        avg_loss = total_loss / len(train_loader)
        if use_wandb:
            wandb.log({"train_loss": avg_loss, "epoch": epochs + 1})
        time_of_day = datetime.datetime.now().strftime("%H:%M:%S")
        total_correct = 0
        total_samples = 0
        model.eval()
        print("Validating..., validate_loader.size", len(validate_loader))
        tot_val_loss = 0
        for i, (batch, labels) in enumerate(validate_loader):
            batch, labels = batch.to(device), labels.to(device)
            input = batch
            if len(input.size()) != 4:
                print("WARNING something wrong with dimensions here")
                print("input.size", input.size())
                continue
            outputs = model(input)
            _, predicted_labels = torch.max(outputs, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            loss = criterion(outputs, labels)
            tot_val_loss += loss.item()
            if i % 1000 == 0:
                time_of_day = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"{time_of_day} Epoch "
                    + str(epochs + 1)
                    + ", iteration "
                    + str(i)
                    + ": "
                    + str(loss.item())
                )
        accuracy = total_correct / total_samples
        loss = tot_val_loss / len(validate_loader)
        if use_wandb:
            wandb.log(
                {
                    "validation_accuracy": accuracy,
                    "epoch": epochs + 1,
                    "validation_loss": loss,
                },
            )
        message = (
            "Validation accuracy after epoch "
            + str(epochs + 1)
            + ": "
            + str(accuracy)
            + " loss: "
            + str(loss)
        )
        if loss < best_loss or accuracy > best_accuracy:
            best_loss = loss
            best_accuracy = accuracy
            folder_name = "cnn"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            torch.save(
                model.state_dict(),
                f"{folder_name}/best_model_loss_cnn_efficientnet_loss{loss}_acc{accuracy}.pth",
            )
            message += " (model saved)"
        print(message)

    model = torch.load(f"best_model_loss_cnn_{best_loss}.pth")
    predicted = []
    actual = []

    total_correct = 0
    total_samples = 0
    for i, (batch, labels) in enumerate(test_loader):
        actual.extend(labels)

        batch, labels = batch.to(device), labels.to(device)
        input = batch.squeeze()
        outputs = model(input).logits
        _, predicted_labels = torch.max(outputs, 1)

        predicted.extend(list(predicted_labels.cpu().numpy()))
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        accuracy = total_correct / total_samples

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
