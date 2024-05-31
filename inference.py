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
def load_data(nr_samples_in_each_class):
    script_path = Path(__file__).resolve().parent
    # data_path = script_path / "training_data"
    data_path = (
        script_path.parent.parent
        / "project"
        / "dataset"
        / "training_data_100ms_noise_50_2"
    )
    cnn_test_dataset = training_pipeline_convolutional.AudioDataSet(
        data_path / "test", nr_samples_in_each_class
    )
    vit_test_dataset = training_pipeline.AudioDataSet(
        data_path / "test", nr_samples_in_each_class
    )
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
        Path(__file__).resolve().parent
        / "cnn"
        / "best_model_loss_cnn_loss1.276456973284775_acc0.6308243727598566_with_aug_trn_all_lrsTrue.pth"
    )
    model.load_state_dict(torch.load(path_to_cnn))
    model.eval()
    return model.to(device)


def load_vit(
    device,
    num_classes,
    model_filename="best_model_loss_vit_1.2879105645690978_acc_0.6368534482758621_train_all_True.pth",
):
    config = transformers.AutoConfig.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    config.num_labels = num_classes  # Update the number of labels
    model = transformers.AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(torch.load(model_filename))
    return model


class Ensemble(nn.Module):
    def __init__(self, cnn, vit, vit_probability=0.9, cnn_probability=0.1):
        # super(Ensemble).__init__()
        super().__init__()
        self.cnn = cnn
        self.vit = vit
        self.softmax = nn.Softmax()
        self.vit_probability = vit_probability
        self.cnn_probability = cnn_probability

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


def test_model(
    model, test_loader_cnn, test_loader_vit, device, model_name, model_filename
):
    total_samples = 0
    tot_val_loss = 0
    total_correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    i = 0
    predicted = []
    actual = []
    outputs_list = []
    for cnn, vit in zip(test_loader_cnn, test_loader_vit):
        # if i > 2:
        #     break
        batch_cnn, labels_cnn = cnn
        batch_vit, labels_vit = vit

        test_loader_cnn, test_loader_vit
        batch_vit, labels_vit = batch_vit.to(device), labels_vit.to(device)
        # input = batch.squeeze()
        batch_cnn, labels_cnn = batch_cnn.to(device), labels_cnn.to(device)
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
        outputs_list.extend(list(outputs.cpu().detach().numpy()))
        actual.extend(list(labels_cnn.cpu().numpy()))
        predicted.extend(list(predicted_labels.cpu().numpy()))
        if i % 10 == 0:
            print("i", i)
            time_of_day = datetime.datetime.now().strftime("%H:%M:%S")
            print(
                f"{time_of_day} Model {model_name} Epoch {str(i + 1)}, iteration {str(i)}/{str(len(test_loader_cnn))}: "
            )
        i += 1
    print("total_correct", total_correct)
    print("total_samples", total_samples)
    accuracy = total_correct / total_samples
    loss = tot_val_loss
    return accuracy, loss, actual, predicted, outputs_list


def generate_confusion_matrix(
    actual, predicted, class_names, model_name, nr_samples_in_each_class, acc
):
    # current_display_labels = set()
    # for label in actual:
    #     current_display_labels.add(class_names[label])
    # for label in predicted:
    #     current_display_labels.add(class_names[label])
    cm = confusion_matrix(actual, predicted)  # normalize="false"
    # )  # Normalize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    num_classes = len(class_names)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    fig = ax.get_figure()
    fig.patch.set_facecolor(
        "none"
    )  # Set the figure's facecolor to 'none' for transparency
    ax.patch.set_facecolor("none")  # Set the axes' facecolor to 'none' for transparency
    disp.plot(ax=ax, cmap="viridis", xticks_rotation="vertical")
    plt.xticks(ticks=np.arange(num_classes), labels=class_names, rotation=90)
    plt.yticks(ticks=np.arange(num_classes), labels=class_names)
    plt.tight_layout()
    plt.savefig(
        f"plots/confusion_matrix_{model_name}_{nr_samples_in_each_class}_acc_{acc}.png",
        transparent=True,
    )  # Save the figure with a transparent background
    # plt.show()


def main(
    nr_samples_in_each_class,
    model_filename="vit/used/best_model_loss_vit_0_acc_0_train_all_True_5.pth",
):
    run_cnn = True
    run_vit = False
    run_ensemble = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # nr_samples_in_each_class = 200
    vit_test_loader, cnn_test_loader, num_classes, class_names = load_data(
        nr_samples_in_each_class
    )
    cnn_model = load_cnn(device, num_classes)
    # vit_model = load_vit(device, num_classes, model_filename=model_filename)
    # ensamble = Ensemble(cnn_model, vit_model)
    if run_cnn:
        cnn_acc, cnn_loss, cnn_actual, cnn_predicted, outputs_list = test_model(
            cnn_model, cnn_test_loader, vit_test_loader, device, "CNN", model_filename
        )
        print(f"Cnn model accuracy: {cnn_acc} and loss: {cnn_loss}")
        generate_confusion_matrix(
            cnn_actual,
            cnn_predicted,
            class_names,
            "CNN",
            nr_samples_in_each_class,
            cnn_acc,
        )
    if run_vit:
        vit_acc, vit_loss, vit_actual, vit_predicted, outputs_list = test_model(
            vit_model, cnn_test_loader, vit_test_loader, device, "VIT", model_filename
        )
        print(f"Vit model accuracy: {vit_acc} and loss: {vit_loss}")
        generate_confusion_matrix(
            vit_actual,
            vit_predicted,
            class_names,
            "VIT",
            nr_samples_in_each_class,
            vit_acc,
        )
    if run_ensemble:
        (
            ensamble_acc,
            ensamble_loss,
            ensamble_actual,
            ensamble_predicted,
            outputs_list,
        ) = test_model(
            ensamble,
            cnn_test_loader,
            vit_test_loader,
            device,
            "Ensemble",
            model_filename,
        )
        print(f"Ensamble model accuracy: {ensamble_acc} and loss: {ensamble_loss}")
        generate_confusion_matrix(
            ensamble_actual,
            ensamble_predicted,
            class_names,
            "Ensemble",
            nr_samples_in_each_class,
            ensamble_acc,
        )
    # save the outputs_list to a file with the model name
    model_filename_split = model_filename.split("/")[-1].split(".")[0]
    output_name = f"cnn/outputs/outputs_list_{model_filename_split}.npy"
    if not os.path.exists("cnn/outputs"):
        os.makedirs("cnn/outputs")
    np.save(output_name, outputs_list)

    # if run_vit and i == 2:
    #     actual_labels_name = "vit/outputs/actual_labels.npy"
    #     np.save(actual_labels_name, vit_actual)


if __name__ == "__main__":
    # for i in range(2, 12):
    # model_filename = "resnet/used_models/best_model_loss_resnet_loss1.71_acc0.6906_no_aug_trn_all_lrsTrue_2.pth."
    model_filename = "resnet/used_models/best_model_loss_resnet_loss1.71_acc0.6906_no_aug_trn_all_lrsTrue_2.pth."
    model_filename = "cnn/used/best_model_loss_cnn_loss1.3424_acc0.6499_with_aug_trn_all_lrsTrue_finetuned_2.pth"
    # model_filename = f"vit/used/best_model_loss_vit_0_acc_0_train_all_True_{i}.pth"
    main(nr_samples_in_each_class=100, model_filename=model_filename)
