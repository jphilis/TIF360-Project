from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
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
import torchaudio.transforms as ta_transforms
import torchvision.transforms.v2 as tv_transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from TrainingPipeline import training_pipeline_convolutional, training_pipeline


# This file should test the model with some data.
# We will use the test set for this purpose.
# We will use both cnn and transformer model in an ensemble.
def load_data():
    script_path = Path(__file__).resolve().parent
    # data_path = script_path / "training_data"
    data_path = (
        script_path.parent.parent
        / "project"
        / "dataset"
        / "training_data_100ms_noise_50_2"
    )
    cnn_test_dataset = training_pipeline_convolutional.AudioDataSet(
        data_path / "test", 10
    )
    vit_test_dataset = training_pipeline.AudioDataSet(data_path / "test", 200)
    num_classes = len(os.listdir(data_path / "test"))
    class_names = [p.stem for p in Path(data_path / "test").glob("*")]
    cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=8, shuffle=False)
    vit_test_loader = DataLoader(vit_test_dataset, batch_size=8, shuffle=False)
    return vit_test_loader, cnn_test_loader, num_classes, class_names


def load_cnn(device, num_classes):
    pretrained_model = torchvision.models.vgg16(
        torchvision.models.VGG16_Weights.DEFAULT
    )
    model = training_pipeline_convolutional.CNN(num_classes, pretrained_model)
    path_to_cnn = (
        Path(__file__).resolve().parent / "cnn" / "best_model_loss_cnn_3.67.pth"
    )
    model.load_state_dict(torch.load(path_to_cnn))
    model.eval()
    return model.to(device)


def load_vit(device, num_classes):
    config = transformers.AutoConfig.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    config.num_labels = num_classes  # Update the number of labels
    model = transformers.AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(torch.load("best_model_loss_1.14_trans.pth"))
    return model


class Ensemble(nn.Module):
    def __init__(self, cnn, vit, vit_probability=0.6, cnn_probability=0.4):
        # super(Ensemble).__init__()
        super().__init__()
        self.cnn = cnn
        self.vit = vit
        self.softmax = nn.Softmax()
        self.vit_probability = 0.6
        self.cnn_probability = 0.4

    def forward(self, x_cnn, x_vit):
        cnn_output = self.softmax(self.cnn(x_cnn))
        vit_output = self.softmax(self.vit(x_vit).logits)
        weighted_average = vit_output * self.vit_probability
        weighted_average += cnn_output * self.cnn_probability
        return weighted_average


def get_loss(outputs, labels, criterion, total_correct, total_samples, tot_val_loss):
    _, predicted_labels = torch.max(outputs, 1)
    total_correct += (predicted_labels == labels).sum().item()
    total_samples += labels.size(0)
    loss = criterion(outputs, labels)
    tot_val_loss += loss.item()
    return total_correct, total_samples, tot_val_loss, predicted_labels


def test_model(model, test_loader_cnn, test_loader_vit, device, model_name):
    total_samples = 0
    tot_val_loss = 0
    total_correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    i = 0
    predicted = []
    actual = []
    for cnn, vit in zip(test_loader_cnn, test_loader_vit):
        batch_cnn, labels_cnn = cnn
        batch_vit, labels_vit = vit

        test_loader_cnn, test_loader_vit
        batch_vit, labels_vit = batch_vit.to(device), labels_vit.to(device)
        # input = batch.squeeze()
        batch_cnn, labels_cnn = batch_cnn.to(device), labels_cnn.to(device)
        if i > 2:
            break
        if model_name == "CNN":
            input = batch_cnn
            labels = labels_cnn
            outputs = model(input)
            total_correct, total_samples, tot_val_loss, predicted_labels = get_loss(
                outputs, labels, criterion, total_correct, total_samples, tot_val_loss
            )
        elif model_name == "VIT":
            input = batch_vit.squeeze()
            labels = labels_vit
            outputs = model(input).logits
            total_correct, total_samples, tot_val_loss, predicted_labels = get_loss(
                outputs, labels, criterion, total_correct, total_samples, tot_val_loss
            )
        else:
            input_cnn = batch_cnn
            input_vit = batch_vit.squeeze()
            outputs = model(input_cnn, input_vit)
            total_correct, total_samples, tot_val_loss, predicted_labels = get_loss(
                outputs,
                labels_cnn,
                criterion,
                total_correct,
                total_samples,
                tot_val_loss,
            )
        actual.extend(list(labels_cnn.cpu().numpy()))
        predicted.extend(list(predicted_labels.cpu().numpy()))
        if i % 100 == 0:
            time_of_day = datetime.datetime.now().strftime("%H:%M:%S")
            print(
                f"{time_of_day} Model {model_name} Epoch {str(i + 1)}, iteration {str(i)}/{str(len(test_loader_cnn))}: "
            )
        i += 1
    print("total_correct", total_correct)
    print("total_samples", total_samples)
    accuracy = total_correct / total_samples
    loss = tot_val_loss
    return accuracy, loss, actual, predicted


def generate_confusion_matrix(actual, predicted, class_names, model_name):
    
    current_display_labels = set()
    for label in actual:
        current_display_labels.add(class_names[label])
    for label in predicted:
        current_display_labels.add(class_names[label])
    cm = confusion_matrix(actual, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=current_display_labels)
    num_classes = len(current_display_labels)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    disp.plot(ax=ax, cmap="viridis", xticks_rotation="vertical")
    plt.xticks(ticks=np.arange(num_classes), labels=current_display_labels, rotation=90)
    plt.yticks(ticks=np.arange(num_classes), labels=current_display_labels)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_test_loader, cnn_test_loader, num_classes, class_names = load_data()
    # cnn_model = load_cnn(device, num_classes)
    vit_model = load_vit(device, num_classes)
    # ensamble = Ensemble(cnn_model, vit_model)
    # cnn_acc, cnn_loss, cnn_actual, cnn_predicted = test_model(
    #     cnn_model, cnn_test_loader, vit_test_loader, device, "CNN"
    # )
    # print(f"Cnn model accuracy: {cnn_acc} and loss: {cnn_loss}")
    vit_acc, vit_loss, vit_actual, vit_predicted = test_model(
        vit_model, cnn_test_loader, vit_test_loader, device, "VIT"
    )
    print(f"Vit model accuracy: {vit_acc} and loss: {vit_loss}")
    # ensamble_acc, ensamble_loss, ensamble_actual, ensamble_predicted = test_model(
    #     ensamble, cnn_test_loader, vit_test_loader, device, "Ensemble"
    # )
    # print(f"Ensamble model accuracy: {ensamble_acc} and loss: {ensamble_loss}")
    # generate_confusion_matrix(cnn_actual, cnn_predicted, class_names, "CNN")
    generate_confusion_matrix(vit_actual, vit_predicted, class_names, "VIT")
    # generate_confusion_matrix(
    #     ensamble_actual, ensamble_predicted, class_names, "Ensemble"
    # )


if __name__ == "__main__":
    main()
