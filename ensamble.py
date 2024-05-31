import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def load_saved_outputs(model_filename):
    outputs_list = np.load(model_filename)
    return torch.tensor(outputs_list)


def normalize_outputs(outputs):
    softmax = torch.nn.Softmax(dim=1)
    return softmax(outputs)


def calculate_metrics(outputs, labels):
    _, predicted_labels = torch.max(outputs, 1)
    predicted_labels = predicted_labels.numpy()
    f1 = f1_score(labels, predicted_labels, average="weighted")
    balanced_acc = balanced_accuracy_score(labels, predicted_labels)
    acc = accuracy_score(labels, predicted_labels)
    return f1, balanced_acc, acc, predicted_labels


def ensemble_outputs(output_list):
    total_outputs = sum(output_list)
    normalized_ensemble = normalize_outputs(total_outputs / len(output_list))
    return normalized_ensemble


def generate_confusion_matrix(actual, predicted, class_names, model_name, acc):
    cm = confusion_matrix(actual, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    num_classes = len(class_names)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    fig = ax.get_figure()
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")
    disp.plot(ax=ax, cmap="viridis", xticks_rotation="vertical")
    plt.xticks(ticks=np.arange(num_classes), labels=class_names, rotation=90)
    plt.yticks(ticks=np.arange(num_classes), labels=class_names)
    plt.tight_layout()
    plt.savefig(
        f"plots/confusion_matrix_{model_name}_acc_{acc:.4f}.png", transparent=True
    )


def main():
    actual_labels = np.load("vit/outputs/actual_labels.npy")
    class_names = ["Class1", "Class2", "Class3"]  # Replace with your actual class names
    model_filenames = [
        "vit/outputs/outputs_list_best_model_loss_vit_0_acc_0_train_all_True_2.npy",
        "vit/outputs/outputs_list_best_model_loss_vit_0_acc_0_train_all_True_3.npy",
        # Add other model filenames as needed
    ]
    all_outputs = [load_saved_outputs(filename) for filename in model_filenames]

    with open("metrics.txt", "w") as f:
        # Calculate metrics for individual models
        for i, outputs in enumerate(all_outputs):
            normalized_outputs = normalize_outputs(outputs)
            f1, balanced_acc, acc, predicted_labels = calculate_metrics(
                normalized_outputs, actual_labels
            )
            f.write(
                f"Model {i+1} - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}\n"
            )
            print(
                f"Model {i+1} - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}"
            )
            generate_confusion_matrix(
                actual_labels, predicted_labels, class_names, f"model_{i+1}", acc
            )

        # Calculate metrics for ensemble models
        for i in range(2, len(all_outputs) + 1):
            ensemble_outputs_list = ensemble_outputs(all_outputs[:i])
            f1, balanced_acc, acc, predicted_labels = calculate_metrics(
                ensemble_outputs_list, actual_labels
            )
            f.write(
                f"Ensemble of first {i} models - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}\n"
            )
            print(
                f"Ensemble of first {i} models - F1: {f1:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Accuracy: {acc:.4f}"
            )
            generate_confusion_matrix(
                actual_labels, predicted_labels, class_names, f"ensemble_{i}", acc
            )


if __name__ == "__main__":
    main()
