from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import os
import transformers
from pathlib import Path
import torchaudio.transforms as ta_transforms
import torchvision.transforms.v2 as tv_transforms
import copy
import datetime
import sys
import random


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
batdetect2_main_dir = os.path.join(parent_dir, "batdetect2-main")
sys.path.append(batdetect2_main_dir)
print("batdetect2_main_dir", batdetect2_main_dir)
from batdetect2 import api

print("parent_dir", parent_dir)
# from batdetect2-main import api

try:
    import wandb

    use_wandb = True
except ImportError:
    use_wandb = False


class AudioDataSet(Dataset):
    def __init__(self, root_dir, file_count_threshold, transform=None):
        """
        Args:
            directory (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.file_count_threshold = file_count_threshold

        # Walk through the root directory to get subdirectories
        for index, (dirpath, dirnames, filenames) in enumerate(os.walk(root_dir)):
            # Ignore the root directory, only process subdirectories
            selected_filenames = []
            if dirpath != root_dir:
                class_file_names = []
                for filename in filenames:
                    if filename.endswith(".wav"):
                        class_file_names.append(os.path.join(dirpath, filename))
                        # self.labels.append(index - 1)  # index - 1 to start labeling from 0
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
                        sample_rate=300000,
                        n_mels=128,
                        f_max=120000,
                        f_min=20000,
                        normalized=True,
                    ),
                    tv_transforms.Lambda(lambda img: img.transpose(1, 2)),
                    tv_transforms.Resize((1024, 128)),
                ]
            )

        # # Walk through the root directory to get subdirectories
        # for index, (dirpath, dirnames, filenames) in enumerate(os.walk(root_dir)):
        #     # Ignore the root directory, only process subdirectories
        #     if dirpath != root_dir:
        #         for filename in filenames:
        #             if filename.endswith(".wav"):
        #                 self.filenames.append(os.path.join(dirpath, filename))
        #                 self.labels.append(
        #                     index - 1
        #                 )  # index - 1 to start labeling from 0

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_path = self.filenames[idx]
        label = self.labels[idx]
        # Load and preprocess the audio file
        # y, sr = torchaudio.load(audio_path)  # Load audio file
        # if self.transform:
        #     y = self.transform(y)

        audio = api.load_audio(audio_path, target_samp_rate=256000)
        spec = api.generate_spectrogram(audio)
        spec = torch.transpose(spec, 2, 3)
        spec = F.interpolate(spec, size=(1024, 128), mode="bilinear")
        # plt.imshow(spec[0, 0, :, :].cpu().numpy())
        # plt.show()
        spec = spec.squeeze(
            0
        )  # This changes shape from [1, 1, 1024, 128] to [1, 1024, 128]
        # print("spec.shape", spec.shape)

        return spec, label


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


def main(
    train_all_layers=True,
    load_model_path="best_model_loss_vit_1.46_no_aug.pth",
):
    # Create the dataset
    script_path = Path(__file__).resolve().parent
    # data_path = script_path / "training_data"
    data_path = script_path.parent.parent / "dataset" / "training_data_100ms_noise_50_2"

    train_dataset = AudioDataSet(data_path / "train", 1000)
    validate_dataset = AudioDataSet(data_path / "validate", 50)
    test_dataset = AudioDataSet(data_path / "test", 2)

    # dataset = AudioDataSet(data_path / "train")

    num_classes = len(os.listdir(data_path / "train"))
    class_names = [p.stem for p in Path(data_path / "train").glob("*")]
    print("Number of classes:", num_classes)
    # total_size = len(dataset)
    # train_size = int(total_size * 0.01)
    # test_size = int(total_size * 0.98)
    # validate_size = total_size - train_size - test_size
    #
    # split the dataset
    # train_dataset, validate_dataset, test_dataset = random_split(
    #     dataset, [train_size, validate_size, test_size]
    # )

    # train_dataset = augment_dataset(
    #     train_dataset, augmentations=["Normal", "FM", "TM", "FTM"]
    # )

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load the configuration of the pre-trained model
    config = transformers.AutoConfig.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    config.num_labels = num_classes  # Update the number of labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Using device:", device)
    model = transformers.AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)
    if train_all_layers:
        model.load_state_dict(torch.load(load_model_path))
        for param in model.parameters():
            param.requires_grad = True
        lr = 0.0001
    else:
        for param in model.audio_spectrogram_transformer.encoder.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Freeze encoder layers
    criterion = torch.nn.CrossEntropyLoss()

    if use_wandb:
        wandb.init(project="TIF360", entity="jonatca")
    # Iterate over the dataloader
    num_epochs = 50
    print("Starting training...")
    best_loss = 1000
    for epochs in range(num_epochs):
        # print("training_loader.size", len(train_loader))
        total_loss = 0
        model.train()  # Set model to training mode
        for i, (batch, labels) in enumerate(train_loader):
            batch, labels = batch.to(device), labels.to(device)
            # print("Batch shape:", batch.shape)
            # print("Labels shape:", labels.shape)
            # print("Unique labels:", torch.unique(labels))
            # Select the first sample from the batch
            input = batch.squeeze()
            if len(input.size()) != 3:
                print("something wrong with dimentions here")
                print("input.size", input.size())
                continue
            logits = model(input).logits
            loss = criterion(logits, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                date_and_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{date_and_time}: Epoch "
                    + str(epochs + 1)
                    + ", iteration "
                    + str(i)
                    + ": "
                    + str(loss.item())
                )
        avg_loss = total_loss / len(train_loader)
        if use_wandb:
            wandb.log({"train_loss": avg_loss, "epoch": epochs + 1})
        time_of_day = datetime.datetime.now().strftime("%H:%M:%S")
        # print(f"{time_of_day} Epoch " + str(epochs + 1) + " avg loss: ", avg_loss)
        total_correct = 0
        total_samples = 0
        model.eval()  # Set model to evaluation mode
        print("Validating..., validate_loader.size", len(validate_loader))
        tot_val_loss = 0
        for i, (batch, labels) in enumerate(validate_loader):
            batch, labels = batch.to(device), labels.to(device)

            input = batch.squeeze()
            if len(input.size()) != 3:
                print("WARNING something wrong with dimentions here")
                print("input.size", input.size())
                continue
            outputs = model(input).logits
            _, predicted_labels = torch.max(outputs, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            loss = criterion(outputs, labels)
            tot_val_loss += loss.item()
            if i % 10 == 0:
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
            "Validation accuracy after epoch " + str(epochs + 1) + ": " + str(accuracy)
        )
        if loss < best_loss:
            best_loss = loss
            torch.save(
                model.state_dict(),
                f"best_model_loss_vit_{loss}_acc_{accuracy}_train_all_{train_all_layers}.pth",
            )
            message += " (model saved)"
        print(message)

    # Load the best model
    model = torch.load(f"best_model_loss_vit_{best_loss}.pth")
    # Save the test labels and predictions so we can make confusion matrix from them
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

    # Save the test labels and predictions
    np.save("test_labels.npy", actual)
    np.save("test_predictions.npy", predicted)

    accuracy = total_correct / total_samples
    message = "Final test accuracy: " + str(accuracy)
    print(message)

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(actual, predicted)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
